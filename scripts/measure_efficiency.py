"""Computational efficiency of the models in the given configs, on one device.

Per config (random inputs of the configured size; no data needed):
- parameters (count_parameters.py convention);
- FLOPs of one forward pass for one image (torch.utils.flop_counter; 1 MAC = 2 FLOPs for
  convolutions and matrix products). Diffusion models also report the per-image inference total
  = forward FLOPs x timesteps x testing.ensemble;
- training: seconds per iteration and peak memory at the configured training batch size, one
  iteration = loss as in training (p_losses / BCE+Dice) + backward + optimizer step (+ EMA update);
  estimated training time per fold = s/iter x iterations per epoch x epochs (excludes data
  loading and checkpoint saving);
- inference: latency of one image at batch size 1 (diffusion: complete sampling of the ensemble)
  and amortized per-image time and peak memory at the configured test batch size.

Runs as the training code does: importing the models disables cuDNN (models/mlffm_segdiff.py);
this is recorded in the report.
"""
import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argument import get_argparser as get_training_argparser, sync_config  # noqa: E402
from common.logging import get_logger  # noqa: E402
from count_parameters import count_parameters  # noqa: E402
from forward.forward_process import ForwardProcess  # noqa: E402
from forward.forward_schedules import ForwardSchedule  # noqa: E402
from loss import p_losses  # noqa: E402
from models import MODELS as DIFFUSION_MODELS  # noqa: E402
from reverse.reverse_process import sample  # noqa: E402
from run_kfold import resolve_task  # noqa: E402
from seg_models import MODELS as SEG_MODELS  # noqa: E402
from seg_training import compute_loss  # noqa: E402
from utils.config import load_config  # noqa: E402

DATASET_SIZES = {"ph2": 200, "isic2018": 2594, "ham10000": 10015}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device, repeats):
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def peak_memory(fn, device):
    if device.type != "cuda":
        fn()
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    sync(device)
    return torch.cuda.max_memory_allocated(device) / 2**30


def count_flops(forward):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:  # torch < 2.1
        return None
    with FlopCounterMode(display=False) as counter:
        forward()
    return counter.get_total_flops()


def build(cfg, task, device):
    models = SEG_MODELS if task == "seg" else DIFFUSION_MODELS
    torch.manual_seed(0)
    return models[cfg["model"]["class"]](**cfg["model"]["params"]).to(device)


def inputs(cfg, batch, device):
    size = cfg["dataset"]["input_size"]
    images = torch.rand(batch, cfg["dataset"]["img_channels"], size, size, device=device)
    masks = (torch.rand(batch, cfg["dataset"]["msk_channels"], size, size, device=device) > 0.5).float()
    return images, masks


def measure(config_path, device, warmup, repeats):
    cfg = load_config(config_path)
    task = resolve_task(cfg, None)
    diffusion = task != "seg"
    torch.backends.cudnn.enabled = bool(cfg.get("run", {}).get("cudnn", False))  # as the config trains
    if diffusion:  # as training.py does (adds training.loss_name etc.)
        cfg = sync_config(cfg, get_training_argparser().parse_args(["-c", str(config_path)]))
    model = build(cfg, task, device)
    train_bs = cfg["data_loader"]["train"]["batch_size"]
    test_bs = cfg["data_loader"]["test"]["batch_size"]
    schedule = ForwardSchedule(**cfg["diffusion"]["schedule"]) if diffusion else None
    steps = schedule.timesteps if diffusion else 1
    ensemble = cfg["testing"]["ensemble"] if diffusion else 1

    def forward(images):
        if not diffusion:
            return model(images)
        x = torch.randn(images.shape[0], 1, *images.shape[2:], device=device)
        t = torch.full((images.shape[0],), steps - 1, device=device, dtype=torch.long)
        return model(x, images, t)

    report = {"config": str(config_path), "model": cfg["model"]["class"], "task": task,
              "cudnn_enabled": torch.backends.cudnn.enabled,
              "dataset": cfg["dataset"]["name"], "input_size": cfg["dataset"]["input_size"],
              "parameters": count_parameters(model)}

    # ---- FLOPs (one image) ----
    model.eval()
    one, _ = inputs(cfg, 1, device)
    with torch.no_grad():
        flops = count_flops(lambda: forward(one))
    report["forward_gflops"] = flops / 1e9 if flops is not None else None
    report["forward_gmacs"] = flops / 2e9 if flops is not None else None
    report["inference_forward_passes_per_image"] = steps * ensemble
    report["inference_gflops_per_image"] = flops * steps * ensemble / 1e9 if flops is not None else None

    # ---- training iteration ----
    model.train()
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = getattr(torch.optim, opt_cfg["name"])(model.parameters(), **opt_cfg["params"])
    ema = None
    if cfg["training"].get("ema", {}).get("use"):
        from ema_pytorch import EMA
        ema = EMA(model=model, **cfg["training"]["ema"]["params"]).to(device)
    images, masks = inputs(cfg, train_bs, device)
    forward_process = ForwardProcess(schedule) if diffusion else None

    def train_step():
        optimizer.zero_grad()
        if diffusion:
            t = torch.randint(1, steps, (train_bs,), device=device).long()
            loss, _ = p_losses(forward_process, model, x_start=masks * 2 - 1, g=images, t=t, cfg=cfg)
        else:
            logits = model(images)
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
            loss, _ = compute_loss(logits, masks, cfg["training"]["loss"])
        loss.backward()
        optimizer.step()
        if ema:
            ema.update()

    for _ in range(warmup):
        train_step()
    iteration = timed(train_step, device, repeats)
    n_train = DATASET_SIZES[cfg["dataset"]["name"].lower()]
    n_train -= math.ceil(n_train / cfg.get("run", {}).get("kfold", {}).get("n_splits", 5))
    iters_per_epoch = math.ceil(n_train / train_bs)
    report["training"] = {
        "batch_size": train_bs, "ema": ema is not None, "seconds_per_iteration": iteration,
        "peak_memory_gb": peak_memory(train_step, device),
        "epochs": cfg["training"]["epochs"], "train_images_per_fold": n_train,
        "iterations_per_epoch": iters_per_epoch,
        "estimated_hours_per_fold": iteration * iters_per_epoch * cfg["training"]["epochs"] / 3600,
    }
    del optimizer, ema, images, masks
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- inference ----
    model.eval()
    batch, _ = inputs(cfg, test_bs, device)
    with torch.no_grad():
        for _ in range(warmup):
            forward(one)
        if diffusion:
            latency = timed(lambda: [sample(schedule, model, one, out_channels=1) for _ in range(ensemble)],
                            device, 1)
        else:
            latency = timed(lambda: forward(one), device, repeats)
        batch_forward = timed(lambda: forward(batch), device, repeats)
        batch_memory = peak_memory(lambda: forward(batch), device)
    report["inference"] = {
        "latency_seconds_batch1": latency,
        "test_batch_size": test_bs,
        "amortized_seconds_per_image": batch_forward * steps * ensemble / test_bs,
        "peak_memory_gb_test_batch": batch_memory,
    }
    return report


def environment(device):
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cudnn_enabled": torch.backends.cudnn.enabled, "device": str(device)}
    if device.type == "cuda":
        env.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda)
    return env


def fmt(value, digits):
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(reports, env):
    lines = [f"Device: {env.get('gpu', env['device'])}, torch {env['torch']}, cuDNN enabled: {env['cudnn_enabled']}", "",
             "| Model | Params (M) | GFLOPs / forward | GFLOPs / image (inference) | Train s/iter (batch) | "
             "Train peak mem (GB) | Est. train h / fold | Latency / image, batch 1 (s) | Amortized s / image | "
             "Test peak mem (GB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        tr, inf = r["training"], r["inference"]
        lines.append(
            f"| {r['model']} | {r['parameters']['total'] / 1e6:.2f} | {fmt(r['forward_gflops'], 2)} | "
            f"{fmt(r['inference_gflops_per_image'], 1)} | {tr['seconds_per_iteration']:.3f} ({tr['batch_size']}) | "
            f"{fmt(tr['peak_memory_gb'], 1)} | {tr['estimated_hours_per_fold']:.1f} | {inf['latency_seconds_batch1']:.3f} | "
            f"{inf['amortized_seconds_per_image']:.4f} | {fmt(inf['peak_memory_gb_test_batch'], 1)} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    device = torch.device(args.device)
    get_logger(filename="efficiency", dir=args.out_dir)  # p_losses logs through this singleton
    env = environment(device)
    reports = []
    for path in args.configs:
        reports.append(measure(path, device, args.warmup, args.repeats))
        print(json.dumps(reports[-1]), flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "efficiency.json").write_text(json.dumps({"environment": env, "records": reports}, indent=2))
    (out / "efficiency.md").write_text(markdown(reports, env))
    print(f"saved: {out / 'efficiency.md'}")


if __name__ == "__main__":
    main()
