# SAM-TVM

TVM Relax frontend for [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything).

The model is re-implemented with `tvm.relax.frontend.nn`, so the same computation can be exported to an IRModule and compiled for different targets/devices. The implementation mirrors the original PyTorch modules and keeps numerical parity with them.

## Project structure

```text
sam_tvm/
  modeling/       # common, transformer, image_encoder, prompt_encoder, mask_decoder, sam
  build_sam.py    # build vit_b / vit_l / vit_h and load torch checkpoints
  runtime.py      # SamVm: image encoder VM + lazy prompt/decoder VMs
  predictor.py    # SamPredictor: image caching, prompt transforms, postprocess
script/
  benchmark.py    # PyTorch vs TVM latency benchmark
docs/env.md       # environment setup
```

## Requirements

This repository does not include the original SAM source tree or model checkpoints.

```bash
git clone https://github.com/facebookresearch/segment-anything.git
pip install -e ./segment-anything
pip install apache-tvm torch torchvision
```

Download a SAM checkpoint from the official repository, for example:

```text
checkpoints/sam_vit_b_01ec64.pth
```

## Quick start

```python
import numpy as np

from sam_tvm.build_sam import build_sam_vit_b
from sam_tvm.predictor import SamPredictor

sam = build_sam_vit_b(checkpoint="checkpoints/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

image = ...  # HWC uint8 image
predictor.set_image(image)

masks, iou, low_res = predictor.predict(
    point_coords=np.array([[x, y]], dtype=np.float32),
    point_labels=np.array([1], dtype=np.int32),
)
```

`box` prompts are also supported as `[x0, y0, x1, y1]` in original image coordinates:

```python
masks, iou, low_res = predictor.predict(
    box=np.array([x0, y0, x1, y1], dtype=np.float32),
)
```

## Tests and benchmark

```bash
python -m sam_tvm.modeling.image_encoder
python -m sam_tvm.modeling.prompt_encoder
python -m sam_tvm.modeling.mask_decoder
python -m sam_tvm.modeling.sam
python -m sam_tvm.predictor
python script/benchmark.py --iters 20 --warmup 5
```

## License

Apache License 2.0. This project contains code derived from Meta's Segment Anything, which is also licensed under Apache 2.0. See `LICENSE` for details.
