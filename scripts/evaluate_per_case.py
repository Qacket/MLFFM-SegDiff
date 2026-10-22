"""Per-case (out-of-fold) evaluation of k-fold checkpoints.

For each fold the fold config is rebuilt as in src/run_kfold.py (same split, including
run.kfold.group_by), the fold's checkpoint is loaded (EMA weights when training.ema.use) and
the held-out fold is predicted exactly as src/testing.py / src/seg_testing.py do. Written to
--out_dir (default: <run.kfold.metrics_dir>/../per_case):
  fold{k}_cases.csv   one row per held-out image; metric conventions in src/per_case_metrics.py
  fold{k}_meta.json   checkpoint and sampling settings of the fold
  fold{k}_preds.npz   dataset index, binary prediction (uint8) and score map (float16):
                      diffusion = ensemble-mean sample in [-1, 1] (threshold 0),
                      direct segmentation = sigmoid probability (threshold 0.5)
  fold{k}_samples.npz with --save_samples (diffusion): every sampled mask (float16, cases x samples x H x W)
  oof_cases.csv       all fold{k}_cases.csv in --out_dir
  summary.json        per-fold and overall mean/std/median of the per-case metrics
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.ham10000 import PrepareHAM10000  # noqa: E402
from datasets.isic2018 import PrepareISIC2018  # noqa: E402
from datasets.ph2 import PreparePH2  # noqa: E402
from forward.forward_schedules import ForwardSchedule  # noqa: E402
from loaders.dataloaders import get_dataloaders  # noqa: E402
from models import MODELS as DIFFUSION_MODELS  # noqa: E402
from per_case_metrics import case_metrics  # noqa: E402
from reverse.reverse_process import sample  # noqa: E402
from run_kfold import build_fold_config, resolve_folds, resolve_task  # noqa: E402
from seg_models import MODELS as SEG_MODELS  # noqa: E402
from utils.config import get_conf_name, get_model_path, load_config  # noqa: E402
from utils.tensors import mean_of_list_of_tensors  # noqa: E402

PREPARERS = {"isic2018": PrepareISIC2018, "ph2": PreparePH2, "ham10000": PrepareHAM10000}
METRICS = ["dice", "jaccard", "precision", "recall", "specificity", "accuracy",
           "hd95", "assd", "bf1", "boundary_precision", "boundary_recall"]
COLUMNS = ["fold", "index", "image_id"] + METRICS + ["gt_area", "pred_area", "tp", "fp", "fn",
                                                     "empty_pred", "empty_gt"]


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config_file", required=True, help="base config given to run_kfold.py")
    parser.add_argument("--folds", type=str, help="comma-separated fold indices (default: all)")
    parser.add_argument("--checkpoint", help='checkpoint path with "{fold}" (default: where run_kfold saved it)')
    parser.add_argument("--out_dir")
    parser.add_argument("--device", help="override run.device")
    parser.add_argument("--seed", type=int, help="sampling seed (default: testing.seed, else 0)")
    parser.add_argument("--bf1_tolerance", type=float, default=2.0, help="Boundary F1 tolerance in pixels")
    parser.add_argument("--clip_x0", action="store_true",
                        help="diffusion: sample with x0 clipping regardless of testing.clip_x0 "
                             "(default output then goes to per_case_x0clip, next to per_case)")
    parser.add_argument("--ensemble", type=int, help="diffusion: number of samples (default: testing.ensemble)")
    parser.add_argument("--save_samples", action="store_true",
                        help="diffusion: also save every sample (scripts/sampling_sensitivity.py)")
    parser.add_argument("--eval_on", help="config of another dataset: evaluate on all of its images "
                                          "(cross-dataset test; needs --out_dir)")
    parser.add_argument("--prior", help="prior file replacing dataset.prior.file (e.g. priors for --eval_on)")
    parser.add_argument("--summary_only", action="store_true",
                        help="only rebuild oof_cases.csv and summary.json from existing fold files")
    parser.add_argument("--refresh_ids", action="store_true",
                        help="with --summary_only: rewrite image_id in the fold files from get_ids()[index]")
    return parser


def default_checkpoint(cfg, task):
    name = cfg["model"]["name"] if task == "seg" else get_conf_name(cfg)
    return get_model_path(name=name, dir=cfg["model"]["save_dir"])


def load_model(cfg, task, checkpoint_path, device):
    models = SEG_MODELS if task == "seg" else DIFFUSION_MODELS
    model = models[cfg["model"]["class"]](**cfg["model"]["params"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ema_cfg = cfg.get("training", {}).get("ema", {})
    if ema_cfg.get("use"):
        from ema_pytorch import EMA
        ema = EMA(model=model, **ema_cfg["params"])
        ema.load_state_dict(checkpoint["ema"])
        model = ema.ema_model
    else:
        model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict(cfg, task, model, images, return_samples=False):
    """Returns (binary prediction, score map), both (B, H, W) on the CPU; with return_samples (diffusion)
    also the samples (B, S, H, W)."""
    if task == "seg":
        logits = model(images)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        score = torch.sigmoid(logits)[:, 0].cpu()
        return score > 0.5, score
    schedule = ForwardSchedule(**cfg["diffusion"]["schedule"])
    clip_x0 = cfg["testing"].get("clip_x0", False)
    samples = [sample(schedule, model, images=images, out_channels=1, clip_x0=clip_x0)[-1][:, :1]
               for _ in range(cfg["testing"]["ensemble"])]
    score = mean_of_list_of_tensors(samples)[:, 0].cpu()
    if return_samples:
        return score > 0, score, torch.stack([x[:, 0].cpu() for x in samples], dim=1)
    return score > 0, score


def evaluate_fold(cfg, task, checkpoint, device, seed, ids, bf1_tolerance, save_samples=False):
    torch.backends.cudnn.enabled = bool(cfg.get("run", {}).get("cudnn", False))  # as in training/testing
    model = load_model(cfg, task, checkpoint, device)
    loader = get_dataloaders(cfg, "te", task="seg" if task == "seg" else None)
    if ids is not None and len(ids) != len(loader.dataset.imgs):
        raise ValueError(f"{len(ids)} image files but {len(loader.dataset.imgs)} cached samples")
    torch.manual_seed(seed)
    rows, indices, preds, scores, all_samples = [], [], [], [], []
    gt_threshold = 0.5 if task == "seg" else 0.0
    for batch in loader:
        if batch["mask"].shape[1] != 1:
            raise ValueError("per-case evaluation expects single-channel masks")
        out = predict(cfg, task, model, batch["image"].to(device), return_samples=save_samples)
        pred, score = out[:2]
        if save_samples:
            all_samples.append(out[2].numpy().astype(np.float16))
        gts = batch["mask"][:, 0] > gt_threshold
        for index, p, g in zip(batch["id"].tolist(), pred.numpy(), gts.numpy()):
            m = case_metrics(p, g, bf1_tolerance=bf1_tolerance)
            rows.append({"fold": cfg["run"]["kfold"]["fold"], "index": index,
                         "image_id": ids[index] if ids is not None else "", **m})
        indices.append(batch["id"].numpy())
        preds.append(pred.numpy().astype(np.uint8))
        scores.append(score.numpy().astype(np.float16))
    samples = np.concatenate(all_samples) if save_samples else None
    return rows, np.concatenate(indices), np.concatenate(preds), np.concatenate(scores), samples


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def describe(rows):
    out = {"n_cases": len(rows),
           "n_empty_pred": sum(r["empty_pred"] == "True" for r in rows)}
    for m in METRICS:
        if any(r.get(m) in (None, "") for r in rows):  # fold files written before the metric existed
            continue
        v = np.array([float(r[m]) for r in rows])
        out[m] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)),
                  "median": float(np.nanmedian(v)), "n_undefined": int(np.isnan(v).sum())}
    tp, fp, fn = (sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn"))
    out["pooled_dice"] = 2 * tp / (2 * tp + fp + fn)
    return out


def summarize(out_dir, n_splits, settings):
    fold_files = sorted(out_dir.glob("fold*_cases.csv"), key=lambda p: int(p.stem[4:-6]))
    per_fold = {p.stem[:-6]: read_csv(p) for p in fold_files}
    all_rows = [r for rows in per_fold.values() for r in rows]
    write_csv(out_dir / "oof_cases.csv", all_rows)
    settings["checkpoints"] = {k: json.loads((out_dir / f"{k}_meta.json").read_text())["checkpoint"]
                               for k in per_fold}
    summary = {"folds": [int(k[4:]) for k in per_fold], "complete": len(per_fold) == n_splits,
               "settings": settings, "overall": describe(all_rows),
               "per_fold": {k: describe(v) for k, v in per_fold.items()}}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    return summary


def main():
    args = get_argparser().parse_args()
    base = load_config(args.config_file)
    task = resolve_task(base, None)
    if args.clip_x0:
        if task == "seg":
            raise ValueError("--clip_x0 only applies to diffusion models")
        base["testing"]["clip_x0"] = True
    if (args.ensemble or args.save_samples) and task == "seg":
        raise ValueError("--ensemble and --save_samples only apply to diffusion models")
    if args.ensemble:
        base["testing"]["ensemble"] = args.ensemble
    if args.eval_on:
        if not args.out_dir:
            raise ValueError("--eval_on needs --out_dir")
        if base["dataset"].get("prior") and not args.prior:
            raise ValueError("the model takes a prior: give --prior for the --eval_on images")
        prior = base["dataset"].get("prior")
        base["dataset"] = load_config(args.eval_on)["dataset"]
        if prior:
            base["dataset"]["prior"] = prior
        base["run"]["kfold"]["all_test"] = True
    if args.prior:
        base["dataset"]["prior"] = {"file": args.prior}
    kfold = base.get("run", {}).get("kfold", {})
    n_splits = kfold.get("n_splits", 5)
    metrics_dir = kfold.get("metrics_dir", "./runs/kfold_metrics")
    out_dir = Path(args.out_dir or Path(metrics_dir).parent / ("per_case_x0clip" if args.clip_x0 else "per_case"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else base.get("testing", {}).get("seed", 0)
    settings = {"config": args.config_file, "task": task, "seed": seed, "bf1_tolerance_px": args.bf1_tolerance,
                "resolution": base["dataset"]["input_size"],
                "evaluated_split": (f"all {base['dataset']['name']} images" if kfold.get("all_test")
                                    else "inner_val" if kfold.get("inner_fold") is not None
                                    else "dev_val" if kfold.get("dev_fraction") else "test"),
                "dev_fraction": kfold.get("dev_fraction"), "eval_on": args.eval_on,
                "prior": base["dataset"].get("prior", {}).get("file")}
    if task != "seg":
        settings.update(ensemble=base["testing"]["ensemble"], clip_x0=base["testing"].get("clip_x0", False))

    if not args.summary_only:
        preparer = PREPARERS[base["dataset"]["name"].lower()]
        ids = preparer(data_dir=base["dataset"]["data_dir"], image_size=base["dataset"]["input_size"]).get_ids() or None
        for fold in resolve_folds(n_splits, args.folds, kfold.get("folds")):
            cfg = build_fold_config(base, fold, n_splits, metrics_dir)
            if args.device:
                cfg["run"]["device"] = args.device
            checkpoint = args.checkpoint.format(fold=fold) if args.checkpoint else default_checkpoint(cfg, task)
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(checkpoint)
            rows, indices, preds, scores, samples = evaluate_fold(
                cfg, task, checkpoint, torch.device(cfg["run"]["device"]), seed, ids, args.bf1_tolerance,
                save_samples=args.save_samples)
            if samples is not None:
                np.savez_compressed(out_dir / f"fold{fold}_samples.npz", index=indices, samples=samples)
            write_csv(out_dir / f"fold{fold}_cases.csv", rows)
            np.savez_compressed(out_dir / f"fold{fold}_preds.npz", index=indices, pred=preds, score=scores)
            (out_dir / f"fold{fold}_meta.json").write_text(json.dumps({"checkpoint": checkpoint, **settings}, indent=4))
            print(f"fold {fold}: {len(rows)} cases, mean Dice {np.mean([r['dice'] for r in rows]):.4f}")

    if args.summary_only and args.refresh_ids:
        ids = PREPARERS[base["dataset"]["name"].lower()](
            data_dir=base["dataset"]["data_dir"], image_size=base["dataset"]["input_size"]).get_ids()
        for path in sorted(out_dir.glob("fold*_cases.csv")):
            rows = read_csv(path)
            for row in rows:
                row["image_id"] = ids[int(row["index"])]
            write_csv(path, rows)
            print(f"refreshed image ids in {path}")
    summary = summarize(out_dir, n_splits, settings)
    print(f"saved: {out_dir / 'summary.json'} (folds {summary['folds']}, complete={summary['complete']})")


if __name__ == "__main__":
    main()
