from __future__ import annotations
import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import tvm

from sam_tvm.build_sam import sam_model_registry
from sam_tvm.predictor import SamPredictor
from sam_tvm.runtime import SamVm
from segment_anything.build_sam import build_sam_vit_b as build_torch_sam_vit_b
from segment_anything.predictor import SamPredictor as TorchSamPredictor


@dataclass
class Stats:
    mean_ms: float
    median_ms: float
    min_ms: float
    p90_ms: float
    std_ms: float

    def __str__(self) -> str:
        return (
            f"mean={self.mean_ms:.3f}, median={self.median_ms:.3f}, "
            f"min={self.min_ms:.3f}, p90={self.p90_ms:.3f}, std={self.std_ms:.3f}"
        )


def benchmark(
    fn: Callable[[], object],
    warmup: int,
    iters: int,
    sync: Callable[[], None],
) -> Stats:
    for _ in range(warmup):
        fn()
        sync()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000.0)

    return Stats(
        mean_ms=float(np.mean(times)),
        median_ms=statistics.median(times),
        min_ms=float(np.min(times)),
        p90_ms=float(np.percentile(times, 90)),
        std_ms=float(np.std(times)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TVM SAM vs PyTorch SAM.")
    parser.add_argument("--checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--kind", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--image-h", type=int, default=256)
    parser.add_argument("--image-w", type=int, default=320)
    parser.add_argument("--num-points", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--skip-torch", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    torch_device = torch.device(args.device)
    if torch_device.type != "cuda":
        raise SystemExit("This benchmark only supports CUDA devices.")
    torch_device_index = torch_device.index if torch_device.index is not None else 0
    tvm_device = tvm.cuda(torch_device_index)

    def sync_torch():
        torch.cuda.synchronize(torch_device)

    def sync_tvm():
        tvm_device.sync()

    def bench_torch(fn: Callable[[], object]) -> Stats:
        with torch.inference_mode():
            return benchmark(fn, args.warmup, args.iters, sync_torch)

    print(f"checkpoint: {checkpoint}")
    print(f"kind:       {args.kind}")
    print(f"device:     {torch_device}")
    print(f"warmup:     {args.warmup}")
    print(f"iters:      {args.iters}")

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    torch_sam = None
    torch_predictor = None
    if not args.skip_torch:
        torch_sam = build_torch_sam_vit_b(checkpoint=str(checkpoint)).eval()
        torch_sam = torch_sam.to(torch_device)
        torch_predictor = TorchSamPredictor(torch_sam)

    tvm_sam = sam_model_registry[args.kind](checkpoint=str(checkpoint))
    start = time.perf_counter()
    sam_vm = SamVm(tvm_sam, target="cuda", device=tvm_device)
    sync_tvm()
    encoder_compile_s = time.perf_counter() - start
    predictor = SamPredictor(sam_vm)

    image_size = sam_vm.image_size
    rows = []

    # ------------------------------------------------------------------
    # Image encoder
    # ------------------------------------------------------------------
    x_np = np.random.rand(1, 3, image_size, image_size).astype("float32")
    x_torch = torch.from_numpy(x_np).to(torch_device)
    x_tvm = tvm.runtime.tensor(x_np, tvm_device)

    if torch_sam is not None:
        with torch.inference_mode():
            torch_features = torch_sam.image_encoder(x_torch)
        sync_torch()
    else:
        torch_features = None

    tvm_features = sam_vm.encode_image(x_tvm)
    sync_tvm()

    torch_stats = None
    if torch_sam is not None:
        torch_stats = bench_torch(lambda: torch_sam.image_encoder(x_torch))
    tvm_stats = benchmark(
        lambda: sam_vm.encode_image(x_tvm),
        args.warmup,
        args.iters,
        sync_tvm,
    )
    rows.append(("image_encoder", torch_stats, tvm_stats))

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------
    boxes = np.array(
        [
            image_size * 0.2,
            image_size * 0.2,
            image_size * 0.8,
            image_size * 0.9,
        ],
        dtype="float32",
    )
    decoder_cases = []
    for num_points in args.num_points:
        point_coords = np.random.rand(1, num_points, 2).astype("float32") * image_size
        point_labels = np.random.randint(0, 2, size=(1, num_points)).astype("int32")
        decoder_cases.append(
            (f"decoder_point_{num_points}", point_coords, point_labels, None, True)
        )
    decoder_cases.append(("decoder_box", None, None, boxes, True))
    decoder_cases.append(
        (
            "decoder_point+box",
            np.random.rand(1, 1, 2).astype("float32") * image_size,
            np.ones((1, 1), dtype="int32"),
            boxes,
            True,
        )
    )

    tvm_first_calls = []
    for name, points, labels, box, multimask_output in decoder_cases:
        if points is not None:
            torch_points = torch.from_numpy(points).to(torch_device)
            torch_labels = torch.from_numpy(labels).to(torch_device)
        else:
            torch_points = None
            torch_labels = None
        torch_box = (
            torch.from_numpy(box).to(torch_device)[None, :] if box is not None else None
        )

        if torch_sam is not None:
            def torch_decode(
                points=torch_points,
                labels=torch_labels,
                box=torch_box,
                multimask_output=multimask_output,
            ):
                point_arg = None
                if points is not None:
                    point_arg = (points, labels)
                sparse, dense = torch_sam.prompt_encoder(
                    points=point_arg,
                    boxes=box,
                    masks=None,
                )
                return torch_sam.mask_decoder(
                    image_embeddings=torch_features,
                    image_pe=torch_sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=multimask_output,
                )

            torch_stats = bench_torch(torch_decode)
        else:
            torch_stats = None

        if points is None:
            points_tvm = np.zeros((1, 0, 2), dtype="float32")
            labels_tvm = np.zeros((1, 0), dtype="int32")
        else:
            points_tvm = points
            labels_tvm = labels
        points_tvm = tvm.runtime.tensor(points_tvm, tvm_device)
        labels_tvm = tvm.runtime.tensor(labels_tvm, tvm_device)
        box_tvm = tvm.runtime.tensor(box, tvm_device) if box is not None else None

        def tvm_decode(
            points_tvm=points_tvm,
            labels_tvm=labels_tvm,
            box_tvm=box_tvm,
            multimask_output=multimask_output,
        ):
            return sam_vm.predict_points(
                tvm_features,
                points_tvm,
                labels_tvm,
                boxes=box_tvm,
                multimask_output=multimask_output,
            )

        start = time.perf_counter()
        tvm_decode()
        sync_tvm()
        first_call_ms = (time.perf_counter() - start) * 1000.0
        tvm_first_calls.append((name, first_call_ms))

        tvm_stats = benchmark(tvm_decode, args.warmup, args.iters, sync_tvm)
        rows.append((name, torch_stats, tvm_stats))

    # ------------------------------------------------------------------
    # Full predictor: set_image and predict
    # ------------------------------------------------------------------
    image = np.random.randint(
        0, 256, size=(args.image_h, args.image_w, 3), dtype=np.uint8
    )

    if torch_predictor is not None:
        torch_stats = bench_torch(lambda: torch_predictor.set_image(image))
    else:
        torch_stats = None
    tvm_stats = benchmark(
        lambda: predictor.set_image(image),
        args.warmup,
        args.iters,
        sync_tvm,
    )
    rows.append(("predictor_set_image", torch_stats, tvm_stats))

    point_coords = np.array([[args.image_w * 0.4, args.image_h * 0.6]], dtype="float32")
    point_labels = np.array([1], dtype="int32")

    if torch_predictor is not None:
        torch_stats = bench_torch(
            lambda: torch_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
            )
        )
    else:
        torch_stats = None

    # 预热，确保 lazy compile 不计入稳态时间
    predictor.predict(point_coords=point_coords, point_labels=point_labels)
    sync_tvm()
    tvm_stats = benchmark(
        lambda: predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
        ),
        args.warmup,
        args.iters,
        sync_tvm,
    )
    rows.append(("predictor_predict_N1", torch_stats, tvm_stats))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\nSteady-state latency (ms):")
    print(f"{'case':<26}{'torch':>12}{'tvm':>12}{'speedup':>10}")
    print("-" * 60)
    for name, torch_stats, tvm_stats in rows:
        torch_text = f"{torch_stats.median_ms:.3f}" if torch_stats is not None else "-"
        tvm_text = f"{tvm_stats.median_ms:.3f}"
        if torch_stats is not None:
            speedup = torch_stats.median_ms / tvm_stats.median_ms
            speedup_text = f"{speedup:.2f}x"
        else:
            speedup_text = "-"
        print(f"{name:<26}{torch_text:>12}{tvm_text:>12}{speedup_text:>10}")

    print("\nDetailed stats (ms):")
    for name, torch_stats, tvm_stats in rows:
        if torch_stats is not None:
            print(f"{name}:")
            print(f"  torch: {torch_stats}")
            print(f"  tvm:   {tvm_stats}")
        else:
            print(f"{name}: tvm: {tvm_stats}")

    print("\nTVM compile / first-call:")
    print(f"  encoder VM build:   {encoder_compile_s:.3f} s")
    for name, first_call_ms in tvm_first_calls:
        print(f"  {name}: {first_call_ms:.3f} ms (compile + run)")


if __name__ == "__main__":
    main()
