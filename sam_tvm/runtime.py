from __future__ import annotations
import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from typing import Optional, Tuple, Dict, List, Union

from .modeling.sam import Sam
from .modeling.image_encoder import ImageEncoderViT
from .modeling.prompt_encoder import PromptEncoder
from .modeling.mask_decoder import MaskDecoder

Array = Union[np.ndarray, tvm.runtime.Tensor]


class SamEncoder(nn.Module):
    """只包含 image encoder。归一化和 padding 在 predictor 的 Python 侧完成。"""

    def __init__(self, image_encoder: ImageEncoderViT) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, image: Tensor) -> Tensor:
        return self.image_encoder(image)


class SamDecoder(nn.Module):
    """point / box 的 prompt encoder + mask decoder。"""

    mask_threshold: float = 0.0

    def __init__(
        self,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        multimask_output: bool = True,
    ) -> None:
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.multimask_output = multimask_output

    def forward(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        boxes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        - image_embeddings: (B, embed_dim, H, W)
        - point_coords: (B, N, 2)
        - point_labels: (B, N)
        - boxes: (B, 4) 或 (B, 0, 4)

        返回:
        - low_res_masks: (B, num_masks, H*4, W*4)
        - iou_predictions: (B, num_masks)
        """
        bs = point_coords.shape[0]
        mask_h, mask_w = self.prompt_encoder.mask_input_size
        mask_inputs = op.zeros((bs, 0, mask_h, mask_w), "float32")

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_inputs=mask_inputs,
        )
        masks, iou_predictions = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )

        if self.multimask_output:
            indices = nn.Tensor.from_const(list(range(1, self.mask_decoder.num_mask_tokens)))
        else:
            indices = nn.Tensor.from_const([0])
        masks = op.take(masks, indices=indices, axis=1)
        iou_predictions = op.take(iou_predictions, indices=indices, axis=1)
        return masks, iou_predictions


class SamVm:
    """把已经构建好的 Sam 封装成可复用的 TVM runtime。"""

    def __init__(
        self,
        sam: Sam,
        target: str = "cuda",
        device: Optional[tvm.runtime.Device] = None,
    ) -> None:
        self.sam = sam
        self.target = target
        self.device = device if device is not None else tvm.cuda(0)

        self.image_size = sam.image_encoder.img_size
        self.embed_dim = sam.prompt_encoder.embed_dim
        self.embedding_size = sam.prompt_encoder.image_embedding_size
        self.mask_input_size = sam.prompt_encoder.mask_input_size
        self.mask_threshold = sam.mask_threshold

        self.pixel_mean = sam.pixel_mean.data.numpy().reshape(-1)
        self.pixel_std = sam.pixel_std.data.numpy().reshape(-1)

        self._param_cache: Dict[int, tvm.runtime.Tensor] = {}
        self._decoder_vms: Dict[Tuple[int, bool, bool], tuple[relax.VirtualMachine, list]] = {}

        # 编译固定的 encoder
        self._build_encoder_vm()

    def _build_module(self, module: nn.Module, spec) -> Tuple[relax.VirtualMachine, List[tvm.runtime.Tensor]]:
        """
        输入一个填充好 param 的 nn.Module，利用指定 spec 进行编译，放入 vm，同时填充 self._param_cache，并且返回其中的 params
        """
        # 对 nn.Module 导出为 IRModule + params 对象
        ir_module, params = module.export_tvm(spec)
        # 编译到 target 并放入 vm，并指定一个 dev
        rt_mod = tvm.compile(ir_module, target=self.target)
        vm = relax.VirtualMachine(rt_mod, self.device)

        # 用 params 的每个参数填充 self._param_cache
        rt_params = []
        for _, param in params:
            key = id(param)
            tensor = self._param_cache.get(key)
            if tensor is None:
                if param.data is None:
                    raise RuntimeError(f"Parameter `{param}` is not bound to concrete data.")
                tensor = tvm.runtime.tensor(param.data.numpy(), self.device)
                self._param_cache[key] = tensor
            rt_params.append(tensor)

        return vm, rt_params
    
    def _build_encoder_vm(self):
        encoder = SamEncoder(self.sam.image_encoder)
        spec = {
            "forward": {
                "image": nn.spec.Tensor([1, 3, self.image_size, self.image_size], "float32"),
            }
        }
        self.encoder_vm, self.encoder_params = self._build_module(encoder, spec)

    def _build_decoder_vm(
        self, num_points: int, has_box: bool, multimask_output: bool
    ) -> Tuple[relax.VirtualMachine, list]:
        key = (num_points, has_box, multimask_output)
        if key not in self._decoder_vms:
            decoder = SamDecoder(
                self.sam.prompt_encoder,
                self.sam.mask_decoder,
                multimask_output=multimask_output,
            )
            spec = {
                "forward": {
                    "image_embeddings": nn.spec.Tensor(
                        [1, self.embed_dim, self.embedding_size[0], self.embedding_size[1]],
                        "float32",
                    ),
                    "point_coords": nn.spec.Tensor([1, num_points, 2], "float32"),
                    "point_labels": nn.spec.Tensor([1, num_points], "int32"),
                    "boxes": nn.spec.Tensor([1, 1 if has_box else 0, 4], "float32"),
                }
            }
            self._decoder_vms[key] = self._build_module(decoder, spec)
        return self._decoder_vms[key]

    def _to_device(self, x) -> tvm.runtime.Tensor:
        if isinstance(x, tvm.runtime.Tensor):
            return x
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.ascontiguousarray(x)
        return tvm.runtime.tensor(x, self.device)

    def encode_image(self, image: Array) -> tvm.runtime.Tensor:
        """
        - image: 已经归一化并 padding 到 (1, 3, img_size, img_size)，类型为 np.ndarray 或 tvm.runtime.Tensor
        - 返回: image_embeddings (1, embed_dim, H, W)，类型为 device 上的 tvm.runtime.Tensor
        """
        image = self._to_device(image)
        return self.encoder_vm["forward"](image, *self.encoder_params)

    def predict_points(
        self,
        image_embeddings: tvm.runtime.Tensor,
        point_coords: Array,
        point_labels: Array,
        boxes: Optional[Array] = None,
        multimask_output: bool = True,
    ) -> Tuple[tvm.runtime.Tensor, tvm.runtime.Tensor]:
        """
        - image_embeddings: encode_image 的输出，类型为 device 上的 tvm.runtime.Tensor
        - point_coords: (1, N, 2) 或 (N, 2)，类型为 np.ndarray 或 tvm.runtime.Tensor
        - point_labels: (1, N) 或 (N,)，类型为 int32 的 np.ndarray 或 tvm.runtime.Tensor
        - boxes: None / (4,) / (1, 4)，XYXY 格式，原图像素坐标

        返回:
        - low_res_masks: (1, num_masks, H*4, W*4)
        - iou_predictions: (1, num_masks)
        两者都是 device 上的 tvm.runtime.Tensor
        """
        if hasattr(point_coords, "shape"):
            num_points = point_coords.shape[-2]
        else:
            raise ValueError("point_coords must have a shape.")

        if not isinstance(point_coords, tvm.runtime.Tensor):
            point_coords = np.asarray(point_coords, dtype="float32")
            if point_coords.ndim == 2:
                point_coords = point_coords[None, :, :]
        if not isinstance(point_labels, tvm.runtime.Tensor):
            point_labels = np.asarray(point_labels, dtype="int32")
            if point_labels.ndim == 1:
                point_labels = point_labels[None, :]

        has_box = boxes is not None
        if has_box:
            if isinstance(boxes, tvm.runtime.Tensor):
                boxes = boxes.numpy()
            boxes = np.asarray(boxes, dtype="float32")
            if boxes.ndim == 1:
                boxes = boxes[None, None, :]
            elif boxes.ndim == 2:
                boxes = boxes[None, :, :]
        else:
            boxes = np.zeros((1, 0, 4), dtype="float32")

        vm, params = self._build_decoder_vm(num_points, has_box, multimask_output)
        return vm["forward"](
            image_embeddings,
            self._to_device(point_coords),
            self._to_device(point_labels),
            self._to_device(boxes),
            *params,
        )
