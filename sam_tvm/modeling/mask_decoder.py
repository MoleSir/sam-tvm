from __future__ import annotations
from tvm.relax.frontend import nn
from tvm import relax
from tvm.relax.frontend.nn import Tensor, op
from typing import Type, Tuple, List
from .common import LayerNorm2d, ConvTranspose2d, Sequential
from .transformer import TwoWayTransformer


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        multimask_output: bool = True,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.multimask_output = multimask_output

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = Sequential(
            ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        - image_embedding: (1, transformer_dim, H, W)
        - image_pe: (1, transformer_dim, H, W)
        - sparse_prompt_embeddings: (batch, num_points, transformer_dim) 稀疏 prompt：点、框的 token（每个点 1 个 token，每个框 2 个）
        - dense_prompt_embeddings: (batch, transformer_dim, H, W) 稠密 prompt：输入掩码经卷积后的特征图；没有掩码 prompt 时是全 0
        - multimask_output 输出 1 个 mask 还是 3 个候选 mask
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # if self.multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        """
        - masks: (batch, num_mask_tokens, H*4, W*4)
        - iou_pred: (batch, num_mask_tokens)
        """
        if self.multimask_output:
            indices = nn.Tensor.from_const(list(range(1, self.num_mask_tokens)))
        else:
            indices = nn.Tensor.from_const([0])

        masks = op.take(masks, indices=indices, axis=1)
        iou_pred = op.take(iou_pred, indices=indices, axis=1)


        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predicts masks. See 'forward' for more details.
        - image_embedding: (1, transformer_dim, H, W)
        - image_pe: (1, transformer_dim, H, W)
        - sparse_prompt_embeddings: (batch, num_points, transformer_dim)
        - dense_prompt_embeddings: (batch, transformer_dim, H, W)
        """
        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        """
        - iou_token = nn.Embedding(1, transformer_dim)：一个 token，维度为 transformer_dim
        - num_mask_tokens = num_multimask_outputs + 1
        - mask_tokens = nn.Embedding(num_mask_tokens, transformer_dim)：num_multimask_outputs + 1 个 token

        iou_token.weight: (1, transformer_dim)
        mask_tokens.weight: (num_multimask_outputs + 1, transformer_dim)

        拼接后：(num_multimask_outputs + 2, transformer_dim)
        再对 batch 维度广播，一张图片配多个 prompt 批次
        (batch, num_mask_tokens + 1, transformer_dim)
        """
        output_tokens = op.concat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = op.broadcast_to(
            op.unsqueeze(output_tokens, 0), 
            (sparse_prompt_embeddings.shape[0], output_tokens.shape[0], output_tokens.shape[1])
        )

        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        """
        (batch, num_multimask_outputs + 2, transformer_dim)
        (batch, num_points, transformer_dim)
        按照 1 维度进行 concat
        """
        tokens = op.concat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        """
        (1, transformer_dim, H, W) repeat => (batch, transformer_dim, H, W)
        (batch, transformer_dim, H, W) 相加
        """
        src = repeat_interleave_int(image_embeddings, tokens.shape[0], 0)
        src = src + dense_prompt_embeddings

        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        """
        同上 src
        """
        pos_src = repeat_interleave_int(image_pe, tokens.shape[0], 0)
        b, c, h, w = src.shape

        # Run the transformer
        """
        - src: (batch, transformer_dim, H, W) 作为图片嵌入
        - pos_src: (batch, transformer_dim, H, W) 作为图片位置编码
        - tokens: (batch, n_points, transformer_dim) 作为 prompt 嵌入
        输出的 hs, src：
        - hs: (batch, n_points, transformer_dim) 更新后的 token
        - src: (batch, HxW, transformer_dim) 更新后的图像特征
        n_points = 1 + num_mask_tokens + n_points
        """
        hs, src = self.transformer(src, pos_src, tokens)
        # iou_token_out = hs[:, 0, :] # (batch, transformer_dim)
        # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] # (batch, self.num_mask_tokens, transformer_dim)
        index_0 = nn.Tensor.from_const([0])
        iou_token_out = op.take(hs, indices=index_0, axis=1)
        iou_token_out = op.squeeze(iou_token_out, axis=1)

        indices = nn.Tensor.from_const(list(range(1, 1 + self.num_mask_tokens)))
        mask_tokens_out = op.take(hs, indices=indices, axis=1)

        # Upscale mask embeddings and predict masks using the mask tokens
        # src = src.transpose(1, 2).view(b, c, h, w)
        
        """
        转为图片格式：(batch, HxW, transformer_dim) => (batch, transformer_dim, H, W)
        """
        src = op.permute_dims(src, [0, 2, 1])
        src = op.reshape(src, (b, c, h, w))

        """
        上采样图像特征。空间分辨率 64×64 → 128×128 → 256×256（4 倍），通道 256 → 32
        (batch, transformer_dim, H, W) => (batch, transformer_dim/8, H*4, W*4)
        """
        upscaled_embedding = self.output_upscaling(src)

        # 处理每个 token
        hyper_in_list: List[Tensor] = []
        for i in range(self.num_mask_tokens):
            """
            遍历每个位置，每个 token，mlp 处理
            (batch, transformer_dim) => (batch, transformer_dim/8)
            """
            index = nn.Tensor.from_const([i])
            mask_token = op.take(mask_tokens_out, indices=index, axis=1)
            mask_token = op.squeeze(mask_token, axis=1)
            mask_token = self.output_hypernetworks_mlps[i](mask_token)
            hyper_in_list.append(mask_token)

        # hyper_in = torch.stack(hyper_in_list, dim=1)
        # (batch, self.num_mask_tokens, transformer_dim/8)
        hyper_in = stack(hyper_in_list, 1)
        b, c, h, w = upscaled_embedding.shape

        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        """
        upscaled_embedding: (batch, transformer_dim/8, H*4, W*4) => (batch, transformer_dim/8, H*4*W*4)
        hyper_in: (batch, self.num_mask_tokens, transformer_dim/8)
        @ 得到 (batch, self.num_mask_tokens, H*4, W*4)
        """
        masks = op.matmul(hyper_in, op.reshape(upscaled_embedding, (b, c, h * w)))
        masks = op.reshape(masks, (b, -1, h, w))

        # Generate mask quality predictions
        # (batch, transformer_dim) => (batch, num_mask_tokens)
        iou_pred = self.iou_prediction_head(iou_token_out)

        """
        - masks: (batch, num_mask_tokens, H*4, W*4)
        - iou_pred: (batch, num_mask_tokens)
        """
        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = op.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = op.sigmoid(x)
        return x


def repeat_interleave_int(x: Tensor, repeats: int, dim: int) -> Tensor:
    # 1. 在 dim 后插入一个新维度
    y = op.unsqueeze(x, dim + 1)
    # 2. 把新维度广播到 repeats 大小
    new_shape = list(x.shape)
    new_shape.insert(dim + 1, repeats)
    y = op.broadcast_to(y, tuple(new_shape))
    # 3. 合并 dim 和 dim+1 两个维度（reshape 回原维度数）
    final_shape = list(x.shape)
    final_shape[dim] = final_shape[dim] * repeats
    return op.reshape(y, tuple(final_shape))


def stack(tensors, dim=0):
    expanded = [op.unsqueeze(t, dim) for t in tensors]
    return op.concat(expanded, dim=dim)


if __name__ == '__main__':
    import tvm
    import torch
    from tvm import relax
    import numpy as np
    from segment_anything.modeling.transformer import TwoWayTransformer as TorchTwoWayTransformer
    from segment_anything.modeling.mask_decoder import MaskDecoder as TorchMaskDecoder

    depth, dim, num_heads, mlp_dim = 2, 64, 8, 128
    height = width = 8
    num_points = 5
    batch = 2

    torch_decoder = TorchMaskDecoder(
        num_multimask_outputs=3,
        transformer=TorchTwoWayTransformer(depth, dim, num_heads, mlp_dim),
        transformer_dim=dim,
        iou_head_depth=3,
        iou_head_hidden_dim=64
    ).eval()

    tvm_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth, dim, num_heads, mlp_dim),
        transformer_dim=dim,
        iou_head_depth=3,
        iou_head_hidden_dim=64
    )

    # 同步模型参数
    torch_state = torch_decoder.state_dict()
    tvm_state = tvm_decoder.state_dict()
    for key, value in torch_state.items():
        if 'output_upscaling' in key:
            key = 'output_upscaling.modules.' + key[17:]
        tvm_state[key].data = value.detach()

    # 导出为 ir module
    spec = {
        'forward': {
            'image_embeddings': nn.spec.Tensor([1, dim, height, width], "float32"),
            'image_pe': nn.spec.Tensor([1, dim, height, width], "float32"),
            'sparse_prompt_embeddings': nn.spec.Tensor([batch, num_points, dim], "float32"),
            'dense_prompt_embeddings': nn.spec.Tensor([batch, dim, height, width], "float32"),
        }
    }
    ir_module, params = tvm_decoder.export_tvm(spec)
    # 编译到 cuda
    rt_mod = tvm.compile(ir_module, target='cuda')
    # 放入虚拟机
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(rt_mod, dev)

    # 随机输入
    image_embedding = np.random.rand(1, dim, height, width).astype("float32")
    image_pe = np.random.rand(1, dim, height, width).astype("float32")
    sparse_prompt_embeddings = np.random.rand(batch, num_points, dim).astype("float32")
    dense_prompt_embeddings = np.random.rand(batch, dim, height, width).astype("float32")

    # torch 推理
    with torch.no_grad():
        torch_decoder = torch_decoder.cuda()
        ref_masks, ref_iou_pred = torch_decoder(
            torch.from_numpy(image_embedding).cuda(),
            torch.from_numpy(image_pe).cuda(),
            torch.from_numpy(sparse_prompt_embeddings).cuda(),
            torch.from_numpy(dense_prompt_embeddings).cuda(),
            multimask_output=True,
        )
        ref_masks, ref_iou_pred = ref_masks.cpu(), ref_iou_pred.cpu()
    
    gpu_params = [tvm.runtime.tensor(p.data.numpy(), dev) for _, p in params]            
    out_masks, out_iou_pred = vm["forward"](
        tvm.runtime.tensor(image_embedding, dev), 
        tvm.runtime.tensor(image_pe, dev), 
        tvm.runtime.tensor(sparse_prompt_embeddings, dev),  
        tvm.runtime.tensor(dense_prompt_embeddings, dev),  
        *gpu_params
    )
    out_masks, out_iou_pred = out_masks.numpy(), out_iou_pred.numpy()
    print(out_masks.shape)
    print(out_iou_pred.shape)

    # 比较结果
    diff_masks = (ref_masks - out_masks).abs().max().item()
    diff_iou_pred = (ref_iou_pred - out_iou_pred).abs().max().item()
    print(
        "TwoWayTransformer CPU parity: "
        f"masks max_abs={diff_masks:.3e}, iou_pred max_abs={diff_iou_pred:.3e}"
    )
    assert np.allclose(ref_masks, out_masks, atol=1e-4, rtol=1e-4), diff_masks
    assert np.allclose(ref_iou_pred, out_iou_pred, atol=1e-4, rtol=1e-4), diff_iou_pred
    print("PASS")
