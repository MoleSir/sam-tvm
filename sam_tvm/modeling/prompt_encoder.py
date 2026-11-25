from __future__ import annotations
import math
from tvm.relax.frontend import nn
from tvm import relax
from tvm.relax.frontend.nn import Tensor, op
from typing import Type, Tuple, Optional
from .common import LayerNorm2d, Sequential, Conv2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        - embed_dim: prompt embedding 维度
        - image_embedding_size: 图片编码后的空间尺寸 (H, W)
        - input_image_size: 输入图片经过 padding 后的尺寸 (H, W)
        - mask_in_chans: 编码输入 mask 时使用的隐藏通道数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # 正/负点 + 2 个框角点
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = Sequential(
            Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        - 返回: (1, embed_dim, embedding_h, embedding_w)
        """
        return op.unsqueeze(self.pe_layer(self.image_embedding_size), 0)

    def _embed_points(
        self,
        points: Tensor,
        labels: Tensor,
        pad: bool,
    ) -> Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            """
            padding_point = torch.zeros((points.shape[0], 1, 2))
            padding_label = -torch.ones((labels.shape[0], 1))
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
            """
            padding_point = op.zeros((points.shape[0], 1, 2), "float32")
            padding_label = nn.Tensor.from_const([[-1]])
            padding_label = op.astype(padding_label, "int32")
            padding_label = op.broadcast_to(padding_label, (labels.shape[0], 1))
            points = op.concat([points, padding_point], dim=1)
            labels = op.concat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        """
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        """
        labels = op.unsqueeze(labels, -1)
        neg_label = nn.Tensor.from_const(-1)
        zero_label = nn.Tensor.from_const(0)
        one_label = nn.Tensor.from_const(1)
        zeros = op.zeros(point_embedding.shape, "float32")

        not_a_point = op.broadcast_to(self.not_a_point_embed.weight, point_embedding.shape)
        point_embedding = op.where(op.equal(labels, neg_label), not_a_point, point_embedding)

        point_0 = op.broadcast_to(self.point_embeddings[0].weight, point_embedding.shape)
        point_1 = op.broadcast_to(self.point_embeddings[1].weight, point_embedding.shape)
        point_embedding = point_embedding + op.where(op.equal(labels, zero_label), point_0, zeros)
        point_embedding = point_embedding + op.where(op.equal(labels, one_label), point_1, zeros)
        return point_embedding

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = op.reshape(boxes, (-1, 2, 2))
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)

        """
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        """
        b = corner_embedding.shape[0]
        c = self.embed_dim
        corner_0 = op.broadcast_to(self.point_embeddings[2].weight, (b, 1, c))
        corner_1 = op.broadcast_to(self.point_embeddings[3].weight, (b, 1, c))
        corner_offset = op.concat([corner_0, corner_1], dim=1)
        return corner_embedding + corner_offset

    def _embed_masks(self, masks: Tensor) -> Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Tensor,
        boxes: Tensor,
        masks: Tensor,
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        这里所有 prompt 的第一个维度都相同，直接用 point 的 batch 即可。
        """
        return points.shape[0]

    def forward(
        self,
        point_coords: Tensor,
        point_labels: Tensor,
        boxes: Tensor,
        mask_inputs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        - point_coords: (batch, num_points, 2)
        - point_labels: (batch, num_points)
        - boxes: (batch, 4)，没有框时用 (batch, 0, 4)
        - mask_inputs: (batch, 1, mask_h, mask_w)，没有 mask 时用 (batch, 0, mask_h, mask_w)

        返回:
        - sparse_embeddings: (batch, N, embed_dim)
        - dense_embeddings: (batch, embed_dim, embed_H, embed_W)
        """
        bs = self._get_batch_size(point_coords, boxes, mask_inputs)
        sparse_embeddings = op.zeros((bs, 0, self.embed_dim), "float32")

        if point_coords.shape[1] > 0:
            # boxes 不存在时需要 pad 一个 label=-1 的点
            point_embeddings = self._embed_points(
                point_coords, point_labels, pad=(boxes.shape[1] == 0)
            )
            sparse_embeddings = op.concat([sparse_embeddings, point_embeddings], dim=1)

        if boxes.shape[1] > 0:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = op.concat([sparse_embeddings, box_embeddings], dim=1)

        if mask_inputs.shape[1] > 0:
            dense_embeddings = self._embed_masks(mask_inputs)
        else:
            """
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
            """
            no_mask = op.reshape(self.no_mask_embed.weight, (1, self.embed_dim, 1, 1))
            dense_embeddings = op.broadcast_to(
                no_mask,
                (bs, self.embed_dim, self.image_embedding_size[0], self.image_embedding_size[1]),
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = nn.Parameter((2, num_pos_feats), "float32")
        self.scale = scale
        self.num_pos_feats = num_pos_feats

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = op.matmul(coords, self.positional_encoding_gaussian_matrix)
        coords = (2 * math.pi) * coords
        # outputs d_1 x ... x d_n x C shape
        sin = op.wrap_nested(relax.op.sin(coords._expr), "sin")
        cos = op.wrap_nested(relax.op.cos(coords._expr), "cos")
        return op.concat([sin, cos], dim=-1)

    def forward(self, size: Tuple[int, int]) -> Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = op.ones((h, w), "float32")
        y_embed = op.cumsum(grid, axis=0) - 0.5
        x_embed = op.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        # pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        coords = op.concat([op.unsqueeze(x_embed, -1), op.unsqueeze(y_embed, -1)], dim=-1)
        pe = self._pe_encoding(coords)
        return op.permute_dims(pe, (2, 0, 1))  # C x H x W

    def forward_with_coords(
        self, coords_input: Tensor, image_size: Tuple[int, int]
    ) -> Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))
        """
        scale = nn.Tensor.from_const([1.0 / image_size[1], 1.0 / image_size[0]])
        coords = coords_input * scale
        return self._pe_encoding(coords)


if __name__ == "__main__":
    import tvm
    import torch
    from tvm import relax
    import numpy as np
    from segment_anything.modeling.prompt_encoder import PromptEncoder as TorchPromptEncoder

    # 模型尺寸
    embed_dim = 32
    image_embedding_size = (4, 4)
    input_image_size = (32, 32)
    mask_in_chans = 8
    batch, num_points = 2, 5

    # 初始化模型
    torch_encoder = TorchPromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=input_image_size,
        mask_in_chans=mask_in_chans,
    ).eval()
    tvm_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=input_image_size,
        mask_in_chans=mask_in_chans,
    )

    # 同步模型参数
    torch_state = torch_encoder.state_dict()
    tvm_state = tvm_encoder.state_dict()
    for key, value in torch_state.items():
        if key.startswith('mask_downscaling.'):
            key = 'mask_downscaling.modules.' + key[len('mask_downscaling.'):]
        tvm_state[key].data = value.detach()

    torch_encoder = torch_encoder.cuda()

    def run_case(
        name,
        point_coords,
        point_labels,
        boxes,
        mask_inputs,
    ):
        spec = {
            "forward": {
                "point_coords": nn.spec.Tensor(list(point_coords.shape), "float32"),
                "point_labels": nn.spec.Tensor(list(point_labels.shape), "int32"),
                "boxes": nn.spec.Tensor(list(boxes.shape), "float32"),
                "mask_inputs": nn.spec.Tensor(list(mask_inputs.shape), "float32"),
            },
            "get_dense_pe": {},
        }
        ir_module, params = tvm_encoder.export_tvm(spec)
        rt_mod = tvm.compile(ir_module, target='cuda')
        dev = tvm.cuda(0)
        vm = relax.VirtualMachine(rt_mod, dev)
        gpu_params = [tvm.runtime.tensor(p.data.numpy(), dev) for _, p in params]

        # torch 推理
        with torch.no_grad():
            ref_sparse, ref_dense = torch_encoder(
                points=(
                    torch.from_numpy(point_coords).cuda(),
                    torch.from_numpy(point_labels).cuda(),
                ),
                boxes=torch.from_numpy(boxes).cuda() if boxes.shape[1] > 0 else None,
                masks=torch.from_numpy(mask_inputs).cuda() if mask_inputs.shape[1] > 0 else None,
            )
            ref_pe = torch_encoder.get_dense_pe()
            ref_sparse, ref_dense, ref_pe = ref_sparse.cpu(), ref_dense.cpu(), ref_pe.cpu()

        # TVM 推理
        out_sparse, out_dense = vm["forward"](
            tvm.runtime.tensor(point_coords, dev),
            tvm.runtime.tensor(point_labels, dev),
            tvm.runtime.tensor(boxes, dev),
            tvm.runtime.tensor(mask_inputs, dev),
            *gpu_params,
        )
        out_pe = vm["get_dense_pe"](*gpu_params)
        out_sparse, out_dense, out_pe = out_sparse.numpy(), out_dense.numpy(), out_pe.numpy()

        # 比较结果
        diff_sparse = (ref_sparse - out_sparse).abs().max().item()
        diff_dense = (ref_dense - out_dense).abs().max().item()
        diff_pe = (ref_pe - out_pe).abs().max().item()
        print(
            f"PromptEncoder {name} CPU parity: "
            f"sparse max_abs={diff_sparse:.3e}, dense max_abs={diff_dense:.3e}, "
            f"pe max_abs={diff_pe:.3e}"
        )
        assert np.allclose(ref_sparse.numpy(), out_sparse, atol=1e-4, rtol=1e-4), diff_sparse
        assert np.allclose(ref_dense.numpy(), out_dense, atol=2e-3, rtol=2e-3), diff_dense
        assert np.allclose(ref_pe.numpy(), out_pe, atol=1e-4, rtol=1e-4), diff_pe

    # 用例 1：points + boxes + masks
    point_coords = np.random.rand(batch, num_points, 2).astype("float32") * 32
    point_labels = np.random.randint(0, 2, size=(batch, num_points)).astype("int32")
    boxes = np.random.rand(batch, 4).astype("float32") * 32
    mask_inputs = np.random.rand(batch, 1, 16, 16).astype("float32")
    run_case("points+boxes+masks", point_coords, point_labels, boxes, mask_inputs)

    # 用例 2：只有 points，boxes / masks 用 0 长度张量表示
    point_labels = np.random.randint(0, 2, size=(batch, num_points)).astype("int32")
    empty_boxes = np.zeros((batch, 0, 4), dtype="float32")
    empty_masks = np.zeros((batch, 0, 16, 16), dtype="float32")
    run_case("points-only", point_coords, point_labels, empty_boxes, empty_masks)

    print("PASS")
