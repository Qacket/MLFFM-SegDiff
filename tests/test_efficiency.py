import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def test_flop_count_of_a_convolution():
    from measure_efficiency import count_flops
    conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
    x = torch.rand(1, 3, 16, 16)
    assert count_flops(lambda: conv(x)) == 2 * 8 * 3 * 3 * 3 * 16 * 16


def small(path, tmp_path, name, **params):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["model"]["params"].update(params)
    for section in cfg["data_loader"].values():
        section["batch_size"] = 2
    if "diffusion" in cfg:
        cfg["diffusion"]["schedule"]["timesteps"] = 4
        cfg["testing"]["ensemble"] = 2
    out = tmp_path / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return str(out)


def test_measure_efficiency_cpu(tmp_path):
    tiny = dict(dim_x=8, dim_g=8, dim_x_mults=[1] * 6, dim_g_mults=[1] * 6, resnet_block_groups=1)
    from test_per_case import nondiffusion_config
    nodiff_config = tmp_path / "nondiffusion.yaml"
    nodiff_config.write_text(yaml.safe_dump(nondiffusion_config(
        yaml.safe_load((ROOT / "configs/paper/isic2018/mlffm.yaml").read_text()))))
    configs = [
        small(ROOT / "configs/paper/isic2018/mlffm.yaml", tmp_path, "diff", **tiny),
        small(nodiff_config, tmp_path, "nodiff", **tiny),
        small(ROOT / "configs/paper/isic2018/unet.yaml", tmp_path, "unet", features=[8, 16]),
    ]
    result = subprocess.run([sys.executable, str(ROOT / "scripts/measure_efficiency.py"), "-c", *configs,
                             "--out_dir", str(tmp_path / "out"), "--device", "cpu", "--warmup", "1", "--repeats", "2"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    report = json.loads((tmp_path / "out/efficiency.json").read_text())
    assert report["environment"]["cudnn_enabled"] is False
    diff, nodiff, unet = report["records"]
    assert diff["inference_forward_passes_per_image"] == 4 * 2
    assert diff["inference_gflops_per_image"] == pytest.approx(diff["forward_gflops"] * 8)
    assert nodiff["inference_gflops_per_image"] == pytest.approx(nodiff["forward_gflops"])
    for r in report["records"]:
        assert r["forward_gflops"] > 0 and r["training"]["seconds_per_iteration"] > 0
        assert r["training"]["peak_memory_gb"] is None  # CPU
        assert r["training"]["train_images_per_fold"] == 2594 - 519
    assert diff["training"]["ema"] and nodiff["training"]["ema"] and not unet["training"]["ema"]
    assert diff["parameters"]["total"] > nodiff["parameters"]["total"]  # time embeddings removed
    assert (tmp_path / "out/efficiency.md").read_text().count("\n| ") == 4  # header + 3 models
    assert not (tmp_path / "log.log").exists()
