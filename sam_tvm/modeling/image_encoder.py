from __future__ import annotations
from tvm.relax.frontend import nn
from tvm import relax
from tvm.relax.frontend.nn import Tensor, op
from typing import Type, Tuple, List, Optional
from .common import LayerNorm2d, MLPBlock, Sequential, Conv2d

from .transformer import TwoWayTransformer


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter((1, img_size // patch_size, img_size // patch_size, embed_dim), "float32")

        self.blocks = nn.ModuleList([])
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = Sequential(
            Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # x = self.neck(x.permute(0, 3, 1, 2))
        x = op.permute_dims(x, [0, 3, 1, 2])
        x = self.neck(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter((2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter((2 * input_size[1] - 1, head_dim))

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x)
        qkv = op.reshape(qkv, (B, H * W, 3, self.num_heads, -1))
        qkv = op.permute_dims(qkv, (2, 0, 3, 1, 4))

        # q, k, v with shape (B * nHead, H * W, C)
        # q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        qkv_split = op.split(qkv, 3, axis=0)
        q = op.reshape(qkv_split[0], (B * self.num_heads, H * W, -1))
        k = op.reshape(qkv_split[1], (B * self.num_heads, H * W, -1))
        v = op.reshape(qkv_split[2], (B * self.num_heads, H * W, -1))

        # attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = op.matmul(q * self.scale, op.permute_dims(k, [0, 2, 1]))

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
    
        # attn = attn.softmax(dim=-1)
        # x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        attn = op.softmax(attn, axis=-1)
        x = op.matmul(attn, v)
        x = op.reshape(x, (B, self.num_heads, H, W, -1))
        x = op.permute_dims(x, (0, 2, 3, 1, 4))
        x = op.reshape(x, (B, H, W, -1))

        x = self.proj(x)

        return x


def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        # x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        # TVM 的 pad 在此处会触发 reduction 调度的编译问题，改用 zeros + concat 手动补齐
        if pad_h > 0:
            x = op.concat([x, op.zeros((B, pad_h, W, C), "float32")], dim=1)
        if pad_w > 0:
            x = op.concat([x, op.zeros((B, H + pad_h, pad_w, C), "float32")], dim=2)
    Hp, Wp = H + pad_h, W + pad_w

    # x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    x = op.reshape(x, (B, Hp // window_size, window_size, Wp // window_size, window_size, C))
    x = op.permute_dims(x, (0, 1, 3, 2, 4, 5))
    windows = op.reshape(x, (-1, window_size, window_size, C))

    return windows, (Hp, Wp)


def window_unpartition(
    windows: Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    # x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    x = op.reshape(windows, (B, Hp // window_size, Wp // window_size, window_size, window_size, -1))
    x = op.permute_dims(x, (0, 1, 3, 2, 4, 5))
    x = op.reshape(x, (B, Hp, Wp, -1))

    if Hp > H or Wp > W:
        x = op.wrap_nested(
            relax.op.strided_slice(x._expr, axes=[1, 2], begin=[0, 0], end=[H, W]),
            "strided_slice",
        )

    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        # rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1)
        rel_pos_resized = op.permute_dims(
            op.reshape(rel_pos, (1, rel_pos.shape[0], -1)), (0, 2, 1)
        )
        # TVM 的 interpolate 只支持 4D/5D，这里把 (1, C, L) 看成 (1, C, 1, L)
        rel_pos_resized = op.unsqueeze(rel_pos_resized, 2)
        rel_pos_resized = op.interpolate(
            rel_pos_resized,
            size=(1, max_rel_dist),
            mode="linear",
            data_layout="NCHW",
        )
        # rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        rel_pos_resized = op.reshape(rel_pos_resized, (-1, max_rel_dist))
        rel_pos_resized = op.permute_dims(rel_pos_resized, (1, 0))
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    # q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    # k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)

    # q_coords: torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    q_coords = op.arange(0, q_size, dtype="float32")
    q_coords = op.reshape(q_coords, (q_size, 1))
    q_coords = q_coords * max(k_size / q_size, 1.0)

    # k_coords: torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    k_coords = op.arange(0, k_size, dtype="float32")
    k_coords = op.reshape(k_coords, (1, k_size))
    k_coords = k_coords * max(q_size / k_size, 1.0)

    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    relative_coords = op.astype(relative_coords, "int32")

    # TVM 不支持直接用张量下标，改用 take
    return op.take(rel_pos_resized, relative_coords, axis=0)


def add_decomposed_rel_pos(
    attn: Tensor,
    q: Tensor,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = op.reshape(q, (B, q_h, q_w, dim))
    rel_h = op.wrap_nested(
        relax.op.einsum([r_q._expr, Rh._expr], "bhwc,hkc->bhwk"),
        "einsum",
    )
    rel_w = op.wrap_nested(
        relax.op.einsum([r_q._expr, Rw._expr], "bhwc,wkc->bhwk"),
        "einsum",
    )

    attn = op.reshape(attn, (B, q_h, q_w, k_h, k_w))
    attn = attn + op.unsqueeze(rel_h, 4) + op.unsqueeze(rel_w, 3)
    attn = op.reshape(attn, (B, q_h * q_w, k_h * k_w))

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        # x = x.permute(0, 2, 3, 1)
        x = op.permute_dims(x, (0, 2, 3, 1))
        return x


if __name__ == "__main__":
    import torch
    import tvm
    from functools import partial
    from tvm import relax
    import numpy as np
    from segment_anything.modeling.image_encoder import ImageEncoderViT as TorchImageEncoderViT

    # 模型尺寸
    img_size = 72
    patch_size = 8
    embed_dim, depth, num_heads = 32, 2, 4
    mlp_ratio, out_chans = 2.0, 16
    window_size, global_attn_indexes = 4, (1,)
    batch = 1

    def make_args():
        return dict(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            use_rel_pos=True,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )

    # 初始化模型
    torch_model = TorchImageEncoderViT(**make_args()).eval()
    tvm_model = ImageEncoderViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        out_chans=out_chans,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_rel_pos=True,
        window_size=window_size,
        global_attn_indexes=global_attn_indexes,
    )

    # 同步模型参数
    torch_state = torch_model.state_dict()
    tvm_state = tvm_model.state_dict()
    for key, value in torch_state.items():
        if key.startswith('neck.'):
            key = 'neck.modules.' + key[len('neck.'):]
        tvm_state[key].data = value.detach()

    # 将 tvm Module：1. 导出为 IRModuel（设备无关）、2. 指定平台编译、3. 插入到虚拟机
    spec = {
        "forward": {
            "x": nn.spec.Tensor([batch, 3, img_size, img_size], "float32"),
        }
    }
    ir_module, params = tvm_model.export_tvm(spec)
    rt_mod = tvm.compile(ir_module, target='cuda')
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(rt_mod, dev)

    # 随机 numpy 输入
    image = np.random.rand(batch, 3, img_size, img_size).astype("float32")

    # torch 推理
    with torch.no_grad():
        torch_model = torch_model.cuda()
        ref = torch_model(torch.from_numpy(image).cuda()).cpu()

    # TVM 推理
    gpu_params = [tvm.runtime.tensor(p.data.numpy(), dev) for _, p in params]
    out = vm["forward"](tvm.runtime.tensor(image, dev), *gpu_params).numpy()

    # 比较结果
    diff = (ref - out).abs().max().item()
    print("ImageEncoderViT CPU parity: " f"max_abs={diff:.3e}")
    # conv 在 TVM 和 cuDNN 下的累加顺序不同，允许略大的误差
    assert np.allclose(ref.numpy(), out, atol=2e-3, rtol=2e-3), diff
    print("PASS")
