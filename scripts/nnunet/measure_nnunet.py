"""Parameters, FLOPs and inference time of a trained nnU-Net v2 model (run in the nnU-Net environment).

Loads --model_folder (e.g. $nnUNet_results/Dataset501_ISIC2018/nnUNetTrainer__nnUNetPlans__2d) with
nnUNetPredictor (--fold, default all; checkpoint_final.pth) and reports for an input of the plans' patch size:
- parameters and GFLOPs of one forward pass (torch.utils.flop_counter; 1 MAC = 2 FLOPs);
- forward passes per image as nnUNetv2_predict runs them: sliding-window positions (1 when the
  image equals the patch) x test-time mirroring (2^axes, on by default);
- latency of one forward at batch 1 and the per-image time it implies, the amortized per-image
  time and peak memory at --batch_size.
Network forward only: preprocessing, resampling and export are not timed. nnU-Net runs with cuDNN
enabled (recorded), unlike scripts/measure_efficiency.py, which runs as the repository code does.
"""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_seconds(fn, device, repeats):
    times = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


@torch.no_grad()
def measure_network(net, input_shape, device, forwards_per_image, batch_size, warmup=3, repeats=20):
    """input_shape = (channels, *spatial) of one patch."""
    from torch.utils.flop_counter import FlopCounterMode
    net = net.to(device).eval()
    one = torch.rand(1, *input_shape, device=device)
    batch = torch.rand(batch_size, *input_shape, device=device)
    with FlopCounterMode(display=False) as counter:
        net(one)
    for _ in range(warmup):
        net(one)
        net(batch)
    latency = _median_seconds(lambda: net(one), device, repeats)
    batch_time = _median_seconds(lambda: net(batch), device, repeats)
    peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        net(batch)
        _sync(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
    return {
        "parameters": sum(p.numel() for p in net.parameters()),
        "forward_gflops": counter.get_total_flops() / 1e9,
        "forwards_per_image": forwards_per_image,
        "inference_gflops_per_image": counter.get_total_flops() * forwards_per_image / 1e9,
        "forward_latency_seconds_batch1": latency,
        "latency_seconds_per_image_batch1": latency * forwards_per_image,
        "batch_size": batch_size,
        "amortized_seconds_per_image": batch_time * forwards_per_image / batch_size,
        "peak_memory_gb_batch": peak,
    }


def sliding_window_positions(image_size, patch_size, step=0.5):
    """Patch positions per image, as nnU-Net computes its sliding-window steps."""
    return math.prod(max(1, math.ceil((i - p) / (p * step)) + 1) for i, p in zip(image_size, patch_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_folder", required=True)
    parser.add_argument("--fold", default="all", help="fold whose checkpoint is loaded: all or 0-4 (same network)")
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs="+", help="default: the plans' patch size")
    parser.add_argument("--disable_mirroring", action="store_true", help="as nnUNetv2_predict --disable_tta")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import nnunetv2

    device = torch.device(args.device)
    predictor = nnUNetPredictor(use_mirroring=not args.disable_mirroring, device=device,
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    fold = int(args.fold) if args.fold.isdigit() else args.fold
    predictor.initialize_from_trained_model_folder(args.model_folder, use_folds=(fold,),
                                                   checkpoint_name="checkpoint_final.pth")
    net = predictor.network
    net.load_state_dict(predictor.list_of_parameters[0])
    patch = [int(p) for p in predictor.configuration_manager.patch_size]
    channels = len(predictor.dataset_json["channel_names"])
    axes = predictor.allowed_mirroring_axes if predictor.use_mirroring else None
    mirrors = 2 ** len(axes) if axes else 1
    positions = sliding_window_positions(args.image_size or patch, patch)

    report = measure_network(net, (channels, *patch), device, positions * mirrors, args.batch_size,
                             repeats=args.repeats)
    report.update(model_folder=args.model_folder, fold=args.fold, patch_size=patch, sliding_window_positions=positions,
                  mirroring_forwards=mirrors, environment={
                      "python": platform.python_version(), "torch": torch.__version__,
                      "nnunetv2": getattr(nnunetv2, "__version__", "see pip show nnunetv2"),
                      "cudnn_enabled": torch.backends.cudnn.enabled, "device": str(device),
                      "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
