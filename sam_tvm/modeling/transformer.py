import math
import sys
from pathlib import Path
from typing import Tuple, Type

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

if __package__:
    from .common import MLPBlock
else:  # allow running this file directly: python sam_tvm/modeling/transformer.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sam_tvm.modeling.common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        - image_embedding: (batch, channel, H, W)
        - image_pe: (batch, channel, H, W)
        - point_embedding: (batch, num_points, channel)
        """

        def flatten2_permute(x: Tensor):
            # 融合最后两个维度，并且交换
            # BxCxHxW -> BxCxHW -> BxHWxC
            x = op.reshape(x, (x.shape[0], x.shape[1], -1))
            x = op.permute_dims(x, [0, 2, 1])
            return x 

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = flatten2_permute(image_embedding)
        image_pe = flatten2_permute(image_pe)

        # Prepare queries
        """
        q: (batch, n_points, C)
        k: (batch, HW, C)
        """
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        # 根据 点 的信息，融合图片信息
        for layer in self.layers:
            # 传入当前新的 qk，点的位置编码总是用原始输入，图的位置用固定的 image_pe
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
    

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        # 对 queries 进行自注意力，qkv 都是 queries
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # 用 query 查询 k/v
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        # query 自注意力
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys  


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = op.matmul(q, op.permute_dims(k, (0, 1, 3, 2)))  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = op.softmax(attn, axis=-1)

        # Get output
        out = op.matmul(attn, v)
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = op.reshape(x, (b, n, num_heads, c // num_heads))
        return op.permute_dims(x, axes=[0, 2, 1, 3])

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return op.reshape(x, (b, n_tokens, n_heads * c_per_head))


if __name__ == "__main__":
    import torch
    import tvm
    from tvm import relax
    import numpy as np
    from segment_anything.modeling.transformer import TwoWayTransformer as TorchTwoWayTransformer

    # 模型尺寸
    depth, dim, num_heads, mlp_dim = 2, 64, 8, 128
    height = width = 8
    num_points = 5

    # 初始化模型
    torch_model = TorchTwoWayTransformer(depth, dim, num_heads, mlp_dim).eval()
    tvm_model = TwoWayTransformer(depth, dim, num_heads, mlp_dim)

    # 同步模型参数
    torch_state = torch_model.state_dict()
    tvm_state = tvm_model.state_dict()
    for key, value in torch_state.items():
        tvm_state[key].data = value.detach()

    # 将 tvm Module：1. 导出为 IRModuel（设备无关）、2. 指定平台编译、3. 插入到虚拟机
    spec = {
        "forward": {
            "image_embedding": nn.spec.Tensor([1, dim, height, width], "float32"),
            "image_pe": nn.spec.Tensor([1, dim, height, width], "float32"),
            "point_embedding": nn.spec.Tensor([1, num_points, dim], "float32"),
        }
    }
    ir_module, params = tvm_model.export_tvm(spec)
    rt_mod = tvm.compile(ir_module, target='cuda')
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(rt_mod, dev)

    # 随机 numpy 输入
    image_embedding = np.random.rand(1, dim, height, width).astype("float32")
    image_pe = np.random.rand(1, dim, height, width).astype("float32")
    point_embedding = np.random.rand(1, num_points, dim).astype("float32")

    # torch 推理
    with torch.no_grad():
        torch_model = torch_model.cuda()
        ref_queries, ref_keys = torch_model(
            torch.from_numpy(image_embedding).cuda(),
            torch.from_numpy(image_pe).cuda(),
            torch.from_numpy(point_embedding).cuda(),
        )
        ref_queries = ref_queries.cpu()
        ref_keys = ref_keys.cpu()
        print(type(ref_queries))
        print(type(ref_keys))

    # TVM 推理
    gpu_params = [tvm.runtime.tensor(p.data.numpy(), dev) for _, p in params]
    out_queries, out_keys = vm["forward"](
        tvm.runtime.tensor(image_embedding, dev), 
        tvm.runtime.tensor(image_pe, dev), 
        tvm.runtime.tensor(point_embedding, dev),  
        *gpu_params
    )
    print(type(out_queries), type(out_keys))
    out_queries, out_keys = out_queries.numpy(), out_keys.numpy()
    print(type(out_queries), type(out_keys))

    # 比较结果
    diff_queries = (ref_queries - out_queries).abs().max().item()
    diff_keys = (ref_keys - out_keys).abs().max().item()
    print(
        "TwoWayTransformer CPU parity: "
        f"queries max_abs={diff_queries:.3e}, keys max_abs={diff_keys:.3e}"
    )
    assert np.allclose(ref_queries, out_queries, atol=1e-4, rtol=1e-4), diff_queries
    assert np.allclose(ref_keys, out_keys, atol=1e-4, rtol=1e-4), diff_keys
    print("PASS")

    