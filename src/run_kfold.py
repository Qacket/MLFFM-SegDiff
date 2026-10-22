import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml

from utils.config import load_config

def get_argparser():
    parser = argparse.ArgumentParser(
        prog="RunKFold",
        description="Run k-fold cross-validation for diffusion or segmentation pipelines",
        epilog="",
    )
    parser.add_argument("-c", "--config_file", type=str, required=True, help="")
    parser.add_argument("--task", type=str, choices=["diffusion", "seg"], help="")
    parser.add_argument("--n_splits", type=int, help="")
    parser.add_argument("--folds", type=str, help="comma-separated fold indices, e.g. 0,1,2,3,4")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="only summarize existing fold metrics, e.g. after running folds in separate processes")
    parser.add_argument("--metrics_dir", type=str, default=None, help="default: run.kfold.metrics_dir, else ./runs/kfold_metrics")
    parser.add_argument("--config_dir", type=str, default=None, help="default: run.kfold.config_dir, else a temporary directory")
    return parser


def resolve_task(config, override):
    if override:
        return override
    if config.get("run", {}).get("task") == "seg":
        return "seg"
    return "diffusion"


def resolve_folds(n_splits, folds_arg, default=None):
    """--folds if given, else run.kfold.folds from the config (`default`), else all folds."""
    if folds_arg:
        return [int(f.strip()) for f in folds_arg.split(",") if f.strip() != ""]
    if default is not None:
        return [int(f) for f in default]
    return list(range(n_splits))


def build_fold_config(base_config, fold, n_splits, metrics_dir):
    cfg = json.loads(json.dumps(base_config))
    cfg.setdefault("run", {})
    cfg["run"].setdefault("kfold", {})
    cfg["run"]["kfold"]["enabled"] = True
    cfg["run"]["kfold"]["n_splits"] = n_splits
    cfg["run"]["kfold"]["fold"] = fold
    cfg["run"]["kfold"].setdefault("shuffle", True)
    cfg["run"]["kfold"].setdefault("seed", 0)
    cfg["run"]["kfold"]["metrics_dir"] = metrics_dir

    base_name = cfg["model"]["name"]
    cfg["model"]["name"] = f"{base_name}_f{fold}"

    if "writer_dir" in cfg["run"]:
        cfg["run"]["writer_dir"] = f"{cfg['run']['writer_dir']}/fold_{fold}"

    return cfg


def write_config(cfg, config_dir, base_name, fold):
    if config_dir is None:
        config_dir = tempfile.mkdtemp(prefix="kfold_")
    Path(config_dir).mkdir(exist_ok=True, parents=True)
    out_path = Path(config_dir) / f"{base_name}_f{fold}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_command(cmd):
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def read_metrics(metrics_dir, model_name):
    metrics_path = Path(metrics_dir) / f"{model_name}_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


def aggregate_metrics(metrics_list):
    keys = sorted({k for m in metrics_list if m for k in m.keys()})
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if m and k in m]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        agg[k] = {"mean": mean, "std": var ** 0.5}
    return agg


def main():
    args = get_argparser().parse_args()
    base_config = load_config(args.config_file)

    task = resolve_task(base_config, args.task)
    kfold_cfg = base_config.get("run", {}).get("kfold", {})
    n_splits = args.n_splits or kfold_cfg.get("n_splits", 5)
    folds = resolve_folds(n_splits, args.folds, kfold_cfg.get("folds"))

    base_name = base_config["model"]["name"]
    metrics_dir = args.metrics_dir or kfold_cfg.get("metrics_dir", "./runs/kfold_metrics")
    config_dir = args.config_dir or kfold_cfg.get("config_dir")
    Path(metrics_dir).mkdir(exist_ok=True, parents=True)

    if task == "seg":
        train_script = "src/seg_training.py"
        test_script = "src/seg_testing.py"
    else:
        train_script = "src/training.py"
        test_script = "src/testing.py"

    all_metrics = []
    for fold in folds:
        cfg = build_fold_config(base_config, fold, n_splits, metrics_dir)
        if not args.aggregate_only:
            cfg_path = write_config(cfg, config_dir, base_name, fold)

            run_command([sys.executable, train_script, "-c", str(cfg_path)])

            if not args.skip_test:
                run_command([sys.executable, test_script, "-c", str(cfg_path)])

        metrics = read_metrics(metrics_dir, cfg["model"]["name"])
        if args.aggregate_only and metrics is None:
            raise FileNotFoundError(f"missing metrics of fold {fold} in {metrics_dir}")
        all_metrics.append(metrics)

    agg = aggregate_metrics(all_metrics)
    agg_path = Path(metrics_dir) / f"{base_name}_kfold_summary.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=4)


if __name__ == "__main__":
    main()
