from __future__ import annotations
from functools import partial
from tvm.relax.frontend import nn
from .modeling.image_encoder import ImageEncoderViT
from .modeling.prompt_encoder import PromptEncoder
from .modeling.mask_decoder import MaskDecoder
from .modeling.transformer import TwoWayTransformer
from .modeling.sam import Sam



def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        load_torch_checkpoint(sam, checkpoint)
    return sam


def load_torch_checkpoint(sam: Sam, checkpoint: str) -> None:
    """
    把原始 SAM 的 torch checkpoint 加载到 TVM 模型中。

    TVM 的 Sequential 比 torch 多一层 `.modules.`，这里统一去掉后与 torch key 对齐。
    `pixel_mean` / `pixel_std` 在 torch 中不是 persistent buffer，不在 checkpoint 里，跳过。
    """
    import torch

    with open(checkpoint, "rb") as f:
        torch_state = torch.load(f, map_location="cpu", weights_only=True)

    tvm_state = sam.state_dict()
    tvm_to_torch = {}
    for tvm_key in tvm_state:
        if tvm_key in ("pixel_mean", "pixel_std"):
            continue
        torch_key = tvm_key.replace(".modules.", ".")
        tvm_to_torch[torch_key] = tvm_key

    missing = [key for key in torch_state if key not in tvm_to_torch]
    unused = [key for key in tvm_to_torch if key not in torch_state]
    if missing or unused:
        raise RuntimeError("Checkpoint key mismatch:\n missing in tvm model: {missing[:10]}\n  unused in checkpoint: {unused[:10]}")

    for torch_key, tvm_key in tvm_to_torch.items():
        tvm_state[tvm_key].data = torch_state[torch_key].detach()


if __name__ == "__main__":
    import numpy as np

    checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    sam = build_sam_vit_b(checkpoint=checkpoint)

    # 验证 checkpoint 中的参数是否全部正确写入 TVM 模型
    import torch

    torch_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tvm_state = sam.state_dict()
    max_diff = 0.0
    num_checked = 0
    for tvm_key, param in tvm_state.items():
        if tvm_key in ("pixel_mean", "pixel_std"):
            continue
        torch_key = tvm_key.replace(".modules.", ".")
        ref = torch_state[torch_key].numpy()
        out = param.data.numpy()
        max_diff = max(max_diff, np.abs(ref - out).max())
        num_checked += 1

    print(f"build_sam_vit_b checkpoint load: {num_checked} params, max_abs={max_diff:.3e}")
    assert max_diff == 0.0
    print("PASS")
