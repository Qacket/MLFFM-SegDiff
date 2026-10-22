"""Count registered model parameters from existing YAMLs; no data or training."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform


def count_parameters(model):
    parameters = list(model.parameters())
    return {
        "total": sum(p.numel() for p in parameters),
        "trainable": sum(p.numel() for p in parameters if p.requires_grad),
        "buffers": sum(b.numel() for b in model.buffers()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", help="Default: all repository YAMLs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import yaml
    import monai
    from models import MODELS as diffusion_models
    from seg_models import MODELS as segmentation_models

    root = Path(__file__).resolve().parents[1]
    registry = {**diffusion_models, **segmentation_models}
    paths = args.config or sorted((root / "configs").rglob("*.yaml"))
    records, cache = [], {}
    torch.set_num_threads(1)
    for path in paths:
        raw = path.read_bytes()
        config = yaml.safe_load(raw)
        name, params = config["model"]["class"], config["model"]["params"]
        signature = json.dumps([name, params], sort_keys=True)
        if signature not in cache:
            torch.manual_seed(0)
            model = registry[name](**params).cpu()
            counts = count_parameters(model)
            # Independently sum registered parameter shapes from state_dict.
            state = model.state_dict()
            assert counts["total"] == sum(state[k].numel() for k, _ in model.named_parameters())
            cache[signature] = counts
            del state, model
            gc.collect()
        record = {
            "config": str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root) else str(path),
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "dataset": config["dataset"]["name"], "model": name,
            "model_params": params, "input_size": config["dataset"]["input_size"],
            **cache[signature],
        }
        records.append(record)
        print(f'{record["config"]}: {record["total"]:,}', flush=True)

    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("models", "seg_models", "blocks", "modules", "utils")
        for p in sorted((root / "src" / folder).rglob("*.py"))
    }
    report = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "monai": monai.__version__, "yaml": yaml.__version__, "device": "cpu"},
        "method": "Unique registered parameter elements; 1 M = 1,000,000. Buffers, optimizer and EMA copies excluded. No forward pass.",
        "source_sha256": source_hashes, "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parameters.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 模型参数量比较 / Model parameter comparison", "",
             "按仓库 YAML 实例化模型，CPU 统计；1 M = 1,000,000。不是 GFLOPs 或推理时间。", "",
             "| Model | ISIC2018 Params (M) | PH2 Params (M) | HAM10000 Params (M) |",
             "|---|---:|---:|---:|"]
    for name in ("MLFFMSegDiff", "DermoSegDiff", "UNet", "GenericUNet", "SwinUNETR"):
        cells = []
        for dataset in ("isic2018", "ph2", "ham10000"):
            values = sorted({r["total"] for r in records if r["model"] == name and r["dataset"] == dataset})
            cells.append(" / ".join(f"{v / 1e6:.3f}" for v in values) or "—")
        lines.append("| " + " | ".join([name, *cells]) + " |")
    lines += ["", "## 配置明细", "", "| Config | Total | Trainable | Buffer elements (excluded) |",
              "|---|---:|---:|---:|"]
    for r in records:
        lines.append(f'| `{r["config"]}` | {r["total"]} | {r["trainable"]} | {r["buffers"]} |')
    lines += ["", "统计包含模型注册的全部参数，即使某个模块可能未参与前向；不包括 buffer、优化器状态、EMA 副本。未加载 checkpoint、未验证前向、未训练。模型具体实现以本仓库为准，不代表同名网络的所有公开版本。", "",
              "环境：`" + json.dumps(report["environment"]) + "`。配置内容、源码 SHA-256 与精确计数见 parameters.json。", ""]
    (args.output_dir / "parameters.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
