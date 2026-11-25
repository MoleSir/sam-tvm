from __future__ import annotations
import numpy as np
from tvm.relax.frontend import nn
from tvm import relax
from tvm.relax.frontend.nn import Tensor, op
from typing import List, Tuple
from .image_encoder import ImageEncoderViT
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        - image_encoder: 把图片编码成 image embedding
        - prompt_encoder: 编码点、框、mask 等 prompt
        - mask_decoder: 根据 image embedding 和 prompt 预测 mask
        - pixel_mean / pixel_std: 图片归一化参数
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.pixel_mean = nn.Parameter((3, 1, 1), "float32")
        self.pixel_std = nn.Parameter((3, 1, 1), "float32")
        self.pixel_mean.data = np.array(pixel_mean, dtype="float32").reshape(3, 1, 1)
        self.pixel_std.data = np.array(pixel_std, dtype="float32").reshape(3, 1, 1)

    def preprocess(self, x: Tensor) -> Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        pad_h = self.image_encoder.img_size - h
        pad_w = self.image_encoder.img_size - w
        if pad_h > 0:
            x = op.concat([x, op.zeros((x.shape[0], x.shape[1], pad_h, w), "float32")], dim=2)
        if pad_w > 0:
            x = op.concat([x, op.zeros((x.shape[0], x.shape[1], h + pad_h, pad_w), "float32")], dim=3)
        return x

    def postprocess_masks(
        self,
        masks: Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> Tensor:
        """
        Remove padding and upscale masks to the original image size.

        - masks: (B, C, H, W)，mask_decoder 输出的低分辨率 mask
        - input_size: 输入图片经过 padding 前的尺寸 (H, W)
        - original_size: 原始图片尺寸 (H, W)
        """
        # masks = F.interpolate(masks, (self.image_encoder.img_size, self.image_encoder.img_size), ...)
        masks = op.interpolate(
            masks,
            size=(self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # masks = masks[..., : input_size[0], : input_size[1]]
        masks = op.wrap_nested(
            relax.op.strided_slice(
                masks._expr,
                axes=[2, 3],
                begin=[0, 0],
                end=[input_size[0], input_size[1]],
            ),
            "strided_slice",
        )
        # masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        masks = op.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks

    def forward(
        self,
        image: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        boxes: Tensor,
        mask_inputs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.

        - image: (1, 3, H, W)
        - point_coords: (batch, num_points, 2)
        - point_labels: (batch, num_points)
        - boxes: (batch, 4)，没有框时用 (batch, 0, 4)
        - mask_inputs: (batch, 1, mask_h, mask_w)，没有 mask 时用 (batch, 0, mask_h, mask_w)

        返回:
        - masks: (batch, num_masks, H, W) 二值 mask
        - iou_predictions: (batch, num_masks)
        - low_res_logits: (batch, num_masks, H_low, W_low)
        """
        input_size = (image.shape[2], image.shape[3])
        original_size = input_size

        input_images = self.preprocess(image)
        image_embeddings = self.image_encoder(input_images)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_inputs=mask_inputs,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=original_size,
        )
        # masks = masks > self.mask_threshold
        threshold = nn.Tensor.from_const(self.mask_threshold)
        masks = op.greater(masks, threshold)

        return masks, iou_predictions, low_res_masks



    


if __name__ == "__main__":
    import tvm
    import torch
    from tvm import relax
    import numpy as np
    from functools import partial
    from segment_anything.modeling.image_encoder import ImageEncoderViT as TorchImageEncoderViT
    from segment_anything.modeling.prompt_encoder import PromptEncoder as TorchPromptEncoder
    from segment_anything.modeling.mask_decoder import MaskDecoder as TorchMaskDecoder
    from segment_anything.modeling.transformer import TwoWayTransformer as TorchTwoWayTransformer
    from segment_anything.modeling.sam import Sam as TorchSam
    from .transformer import TwoWayTransformer

    # 模型尺寸
    img_size = 64
    patch_size = 8
    embed_dim, depth, num_heads = 32, 2, 4
    prompt_embed_dim = 16
    image_embedding_size = (img_size // patch_size, img_size // patch_size)
    mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
    num_points = 5
    point_batch = 2
    image_h = image_w = 48

    # 初始化 torch 模型
    torch_sam = TorchSam(
        image_encoder=TorchImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=2.0,
            out_chans=prompt_embed_dim,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            use_rel_pos=True,
            window_size=4,
            global_attn_indexes=(1,),
        ),
        prompt_encoder=TorchPromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=(img_size, img_size),
            mask_in_chans=8,
        ),
        mask_decoder=TorchMaskDecoder(
            num_multimask_outputs=3,
            transformer=TorchTwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=32,
                num_heads=4,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=32,
        ),
    ).eval()

    # 初始化 tvm 模型
    tvm_sam = Sam(
        image_encoder=ImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=2.0,
            out_chans=prompt_embed_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_rel_pos=True,
            window_size=4,
            global_attn_indexes=(1,),
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=(img_size, img_size),
            mask_in_chans=8,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=32,
                num_heads=4,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=32,
        ),
    )

    # 同步模型参数
    torch_state = torch_sam.state_dict()
    tvm_state = tvm_sam.state_dict()
    for key, value in torch_state.items():
        tvm_key = key.replace('.modules.', '.')
        tvm_key = tvm_key.replace('image_encoder.neck.', 'image_encoder.neck.modules.')
        tvm_key = tvm_key.replace(
            'prompt_encoder.mask_downscaling.', 'prompt_encoder.mask_downscaling.modules.'
        )
        tvm_key = tvm_key.replace(
            'mask_decoder.output_upscaling.', 'mask_decoder.output_upscaling.modules.'
        )
        tvm_state[tvm_key].data = value.detach()

    # 导出为 ir module
    spec = {
        'forward': {
            'image': nn.spec.Tensor([1, 3, image_h, image_w], "float32"),
            'point_coords': nn.spec.Tensor([point_batch, num_points, 2], "float32"),
            'point_labels': nn.spec.Tensor([point_batch, num_points], "int32"),
            'boxes': nn.spec.Tensor([point_batch, 0, 4], "float32"),
            'mask_inputs': nn.spec.Tensor([point_batch, 0, mask_input_size[0], mask_input_size[1]], "float32"),
        }
    }
    ir_module, params = tvm_sam.export_tvm(spec)
    # 编译到 cuda
    rt_mod = tvm.compile(ir_module, target='cuda')
    # 放入虚拟机
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(rt_mod, dev)

    # 随机输入
    image = np.random.rand(1, 3, image_h, image_w).astype("float32")
    point_coords = np.random.rand(point_batch, num_points, 2).astype("float32") * img_size
    point_labels = np.random.randint(0, 2, size=(point_batch, num_points)).astype("int32")
    boxes = np.zeros((point_batch, 0, 4), dtype="float32")
    mask_inputs = np.zeros((point_batch, 0, mask_input_size[0], mask_input_size[1]), dtype="float32")

    # torch 推理
    with torch.no_grad():
        torch_sam = torch_sam.cuda()
        ref = torch_sam(
            batched_input=[
                {
                    "image": torch.from_numpy(image[0]).cuda(),
                    "original_size": (image_h, image_w),
                    "point_coords": torch.from_numpy(point_coords).cuda(),
                    "point_labels": torch.from_numpy(point_labels).cuda(),
                }
            ],
            multimask_output=True,
        )[0]
        ref_masks = ref["masks"].cpu()
        ref_iou_pred = ref["iou_predictions"].cpu()
        ref_low_res = ref["low_res_logits"].cpu()

    # TVM 推理
    gpu_params = [tvm.runtime.tensor(p.data.numpy(), dev) for _, p in params]
    out_masks, out_iou_pred, out_low_res = vm["forward"](
        tvm.runtime.tensor(image, dev),
        tvm.runtime.tensor(point_coords, dev),
        tvm.runtime.tensor(point_labels, dev),
        tvm.runtime.tensor(boxes, dev),
        tvm.runtime.tensor(mask_inputs, dev),
        *gpu_params
    )
    out_masks = out_masks.numpy()
    out_iou_pred = out_iou_pred.numpy()
    out_low_res = out_low_res.numpy()

    # 比较结果
    diff_masks = np.abs(ref_masks.numpy().astype("float32") - out_masks.astype("float32")).max()
    mismatch_masks = np.mean(ref_masks.numpy() != out_masks)
    diff_iou_pred = (ref_iou_pred - out_iou_pred).abs().max().item()
    diff_low_res = (ref_low_res - out_low_res).abs().max().item()
    print(
        "Sam CPU parity: "
        f"masks max_abs={diff_masks:.3e}, masks mismatch={mismatch_masks:.3e}, "
        f"iou_pred max_abs={diff_iou_pred:.3e}, "
        f"low_res max_abs={diff_low_res:.3e}"
    )
    # 阈值化对 low_res 的微小误差很敏感，允许极少量边界像素不同
    assert mismatch_masks <= 1e-3, mismatch_masks
    assert np.allclose(ref_iou_pred.numpy(), out_iou_pred, atol=2e-3, rtol=2e-3), diff_iou_pred
    assert np.allclose(ref_low_res.numpy(), out_low_res, atol=2e-3, rtol=2e-3), diff_low_res
    print("PASS")
