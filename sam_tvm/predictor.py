from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union

from .modeling.sam import Sam
from .runtime import SamVm
from .utils.transforms import ResizeLongestSide


def _resize_bilinear(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """把 (B, C, H, W) 的 float 数组双线性插值到 (out_h, out_w)，align_corners=False。"""
    in_h, in_w = x.shape[-2:]
    if in_h == out_h and in_w == out_w:
        return x

    scale_h = in_h / out_h
    scale_w = in_w / out_w

    src_y = (np.arange(out_h, dtype=np.float64) + 0.5) * scale_h - 0.5
    src_x = (np.arange(out_w, dtype=np.float64) + 0.5) * scale_w - 0.5

    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    wy = (src_y - y0).astype(np.float32).reshape(1, 1, out_h, 1)
    wx = (src_x - x0).astype(np.float32).reshape(1, 1, 1, out_w)

    y1 = y0 + 1
    x1 = x0 + 1
    y0 = np.clip(y0, 0, in_h - 1)
    y1 = np.clip(y1, 0, in_h - 1)
    x0 = np.clip(x0, 0, in_w - 1)
    x1 = np.clip(x1, 0, in_w - 1)

    top = x[:, :, y0, :]
    bottom = x[:, :, y1, :]
    out = top * (1 - wy) + bottom * wy

    left = out[:, :, :, x0]
    right = out[:, :, :, x1]
    out = left * (1 - wx) + right * wx
    return out.astype(np.float32, copy=False)


class SamPredictor:
    def __init__(
        self,
        sam_model: Union[Sam, SamVm],
        target: str = "cuda",
        device=None,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        - sam_model: 直接传入 Sam 即可；内部会自动封装成 SamVm（对用户隐藏 VM 细节）
        - sam_model 也可以直接是已经构建好的 SamVm
        """
        super().__init__()
        if isinstance(sam_model, SamVm):
            self.model = sam_model
        else:
            self.model = SamVm(sam_model, target=target, device=device)
        self.transform = ResizeLongestSide(self.model.image_size)
        self.pixel_mean = np.asarray(self.model.pixel_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.pixel_std = np.asarray(self.model.pixel_std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.reset_image()
        
    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        - image: HWC uint8 图片，像素值范围 [0, 255]
        - image_format: "RGB" 或 "BGR"
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.sam.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image = np.asarray(input_image, dtype=np.float32).transpose(2, 0, 1)[None, :, :, :]

        self.reset_image()
        self.original_size = image.shape[:2]
        self.input_size = tuple(input_image.shape[-2:])
        self.features = self.model.encode_image(self._preprocess(input_image))
        self.is_image_set = True

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """Normalize pixel values and pad to a square input."""
        x = x.astype(np.float32, copy=False)
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        pad_h = self.model.image_size - h
        pad_w = self.model.image_size - w
        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        return np.ascontiguousarray(x, dtype=np.float32)

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.
        当前版本支持 point 和单个 box prompt（XYXY，原图像素坐标）。
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        if mask_input is not None:
            raise NotImplementedError("Current predictor does not support mask_input yet.")
        if point_coords is None and box is None:
            raise ValueError("At least one of point_coords or box must be supplied.")

        # Transform input prompts
        if point_coords is not None:
            assert point_labels is not None, (
                "point_labels must be supplied if point_coords is supplied."
            )
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_np = np.asarray(point_coords, dtype=np.float32)[None, :, :]
            labels_np = np.asarray(point_labels, dtype=np.int32)[None, :]
        else:
            coords_np = np.zeros((1, 0, 2), dtype=np.float32)
            labels_np = np.zeros((1, 0), dtype=np.int32)

        boxes_np = None
        if box is not None:
            box = np.asarray(box, dtype=np.float32).reshape(-1)
            assert box.shape == (4,), f"box must have shape (4,), got {box.shape}."
            boxes_np = self.transform.apply_boxes(box, self.original_size)

        low_res_masks_runtime, iou_predictions_runtime = self.model.predict_points(
            self.features,
            coords_np,
            labels_np,
            boxes=boxes_np,
            multimask_output=multimask_output,
        )
        low_res_masks = low_res_masks_runtime.numpy()
        iou_predictions = iou_predictions_runtime.numpy()

        # Upscale the masks to the original image resolution
        masks = self._postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks[0], iou_predictions[0], low_res_masks[0]

    def _postprocess_masks(
        self,
        masks: np.ndarray,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Remove padding and upscale masks to the original image size.
        """
        masks = _resize_bilinear(masks, self.model.image_size, self.model.image_size)
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = _resize_bilinear(masks, original_size[0], original_size[1])
        return masks

    def get_image_embedding(self) -> np.ndarray:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM.
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features.numpy()

    @property
    def device(self):
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None


if __name__ == "__main__":
    import torch
    from .build_sam import build_sam_vit_b
    from segment_anything.build_sam import build_sam_vit_b as build_torch_sam_vit_b
    from segment_anything.predictor import SamPredictor as TorchSamPredictor

    checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    image_h, image_w = 256, 320

    # 直接复用 build_sam
    torch_sam = build_torch_sam_vit_b(checkpoint=checkpoint).eval()
    tvm_sam = build_sam_vit_b(checkpoint=checkpoint)

    # 构建 predictor（对用户隐藏 SamVm）
    torch_predictor = TorchSamPredictor(torch_sam.cuda())
    predictor = SamPredictor(tvm_sam)

    # set_image 对比
    image = np.random.randint(0, 256, size=(image_h, image_w, 3), dtype=np.uint8)
    torch_predictor.set_image(image)
    predictor.set_image(image)
    ref_features = torch_predictor.get_image_embedding().detach().cpu().numpy()
    out_features = predictor.get_image_embedding()
    diff_features = np.abs(ref_features - out_features).max()
    print(f"Predictor image embedding parity: max_abs={diff_features:.3e}")
    assert np.allclose(ref_features, out_features, atol=5e-3, rtol=5e-3), diff_features

    def check_predict(name, point_coords=None, point_labels=None, box=None, multimask_output=True):
        with torch.no_grad():
            ref_masks, ref_iou, ref_low_res = torch_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )
        out_masks, out_iou, out_low_res = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )
        diff_masks = np.abs(ref_masks.astype("float32") - out_masks.astype("float32")).max()
        mismatch_masks = np.mean(ref_masks != out_masks)
        diff_iou = np.abs(ref_iou - out_iou).max()
        diff_low_res = np.abs(ref_low_res - out_low_res).max()
        print(
            f"Predictor {name} parity: "
            f"masks max_abs={diff_masks:.3e}, masks mismatch={mismatch_masks:.3e}, "
            f"iou max_abs={diff_iou:.3e}, "
            f"low_res max_abs={diff_low_res:.3e}"
        )
        # 阈值化对 low_res 的微小误差很敏感，允许极少量边界像素不同
        assert mismatch_masks <= 1e-3, mismatch_masks
        assert np.allclose(ref_iou, out_iou, atol=2e-3, rtol=2e-3), diff_iou
        assert np.allclose(ref_low_res, out_low_res, atol=2e-2, rtol=2e-2), diff_low_res

    # N=1
    point_coords = np.array([[image_w * 0.4, image_h * 0.6]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    check_predict("N=1", point_coords, point_labels)

    # N=1, single mask
    check_predict("N=1 single-mask", point_coords, point_labels, multimask_output=False)

    # N=3，触发新的 decoder 编译
    point_coords = np.array(
        [
            [image_w * 0.3, image_h * 0.4],
            [image_w * 0.7, image_h * 0.3],
            [image_w * 0.5, image_h * 0.8],
        ],
        dtype=np.float32,
    )
    point_labels = np.array([1, 0, 1], dtype=np.int32)
    check_predict("N=3", point_coords, point_labels)

    # box-only
    box = np.array([image_w * 0.2, image_h * 0.2, image_w * 0.8, image_h * 0.9], dtype=np.float32)
    check_predict("box-only", box=box)

    # points + box
    point_coords = np.array([[image_w * 0.4, image_h * 0.6]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    check_predict("points+box", point_coords, point_labels, box=box)

    print("PASS")
