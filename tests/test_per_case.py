import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from per_case_metrics import boundary, case_metrics  # noqa: E402
from utils.kfold import get_kfold_indices  # noqa: E402


def square(shift=0, size=64):
    m = np.zeros((size, size), bool)
    m[10:30, 10 + shift:30 + shift] = True
    return m


def test_identical_and_shifted_squares():
    m = case_metrics(square(), square())
    assert (m["dice"], m["jaccard"], m["hd95"], m["assd"], m["bf1"]) == (1.0, 1.0, 0.0, 0.0, 1.0)
    m = case_metrics(square(3), square(), bf1_tolerance=3)
    assert m["dice"] == pytest.approx(2 * 17 * 20 / 800)
    assert m["jaccard"] == pytest.approx(17 * 20 / (23 * 20))
    assert 0 < m["hd95"] <= 3 and m["bf1"] == 1.0
    assert 0 < m["assd"] <= m["hd95"]
    assert case_metrics(square(3), square(), bf1_tolerance=1)["bf1"] < 1.0


def test_empty_masks():
    empty = np.zeros((64, 64), bool)
    m = case_metrics(empty, empty)
    assert (m["dice"], m["hd95"], m["assd"], m["bf1"]) == (1.0, 0.0, 0.0, 1.0)
    m = case_metrics(empty, square())
    assert (m["dice"], m["bf1"], m["empty_pred"]) == (0.0, 0.0, True)
    assert m["hd95"] == m["assd"] == pytest.approx(np.hypot(64, 64)) and np.isnan(m["precision"])
    assert case_metrics(square(), empty)["empty_gt"]


def test_hd95_and_assd_match_brute_force():
    rng = np.random.default_rng(0)
    from scipy import ndimage
    for _ in range(5):
        a = ndimage.gaussian_filter(rng.random((48, 48)), 3) > 0.5
        b = ndimage.gaussian_filter(rng.random((48, 48)), 3) > 0.5
        assert a.any() and b.any()
        pa, pb = np.argwhere(boundary(a)), np.argwhere(boundary(b))
        d = np.linalg.norm(pa[:, None] - pb[None], axis=-1)
        distances = np.concatenate([d.min(1), d.min(0)])
        m = case_metrics(a, b)
        assert m["hd95"] == pytest.approx(np.percentile(distances, 95))
        assert m["assd"] == pytest.approx(distances.mean())


def small_config(cfg, tmp_path, name):
    out = tmp_path / name
    cfg["run"].update(device="cpu", writer_dir=str(out / "runs"), log_dir=str(out / "logs"))
    cfg["run"]["kfold"] = {"n_splits": 5, "shuffle": True, "seed": 0,
                           "metrics_dir": str(out / "metrics"), "config_dir": str(out / "configs")}
    cfg["model"]["save_dir"] = str(out / "models")
    cfg["model"]["params"].update(dim_x=8, dim_g=8, dim_x_mults=[1] * 6, dim_g_mults=[1] * 6, resnet_block_groups=1)
    cfg["dataset"]["data_dir"] = str(tmp_path / "data")
    cfg["augmentation"]["p"] = 0
    cfg["training"]["epochs"] = 1
    cfg["training"]["ema"]["params"]["update_after_step"] = 0
    for section in cfg["data_loader"].values():
        section.update(batch_size=2, num_workers=0, pin_memory=False)
    if "diffusion" in cfg:
        cfg["diffusion"]["schedule"]["timesteps"] = 10
        cfg["training"]["intial_weights"]["use"] = False
        cfg["testing"]["ensemble"] = 2
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, out


def run(cmd, tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


def nondiffusion_config(cfg):
    """The same network without diffusion (MLFFMSeg, BCE + Dice), as for the PH2 comparison in the paper."""
    cfg["run"]["task"] = "seg"
    cfg["model"].update({"name": "mlffm_nodiff", "class": "MLFFMSeg"})
    del cfg["diffusion"]
    cfg["training"]["loss"] = {"bce": {"coefficient": 1.0}, "dice": {"coefficient": 1.0}}
    cfg["testing"] = {"model_weights": {"overload": False, "file_path": ""}}
    return cfg


@pytest.mark.parametrize("arm", ["diffusion", "nondiffusion"])
def test_per_case_reproduces_testing(arm, tmp_path):
    """Per-case evaluation of a fold checkpoint gives the same pooled Dice as testing.py / seg_testing.py."""
    cfg = yaml.safe_load((ROOT / "configs/paper/isic2018/mlffm.yaml").read_text())
    if arm == "nondiffusion":
        cfg = nondiffusion_config(cfg)
    else:
        cfg["testing"]["seed"] = 0  # testing.py samples with this seed; evaluate_per_case.py defaults to 0
    config, out = small_config(cfg, tmp_path, arm)
    n = 10
    ids = [f"ISIC_{i:07d}" for i in range(n)]
    cache = tmp_path / "data/np"
    cache.mkdir(parents=True)
    images = tmp_path / "data/ISIC2018_Task1-2_Training_Input"
    images.mkdir()
    for image_id in ids:
        (images / f"{image_id}.jpg").touch()
    rng = np.random.default_rng(4)
    np.save(cache / "X_tr_128x128.npy", rng.random((n, 3, 128, 128), dtype=np.float32))
    y = np.zeros((n, 1, 128, 128), np.float32)
    y[:, :, 30:90, 40:100] = 1
    np.save(cache / "Y_tr_128x128.npy", y)

    run([sys.executable, "src/run_kfold.py", "-c", str(config), "--folds", "0"], tmp_path)
    run([sys.executable, "scripts/evaluate_per_case.py", "-c", str(config), "--folds", "0"], tmp_path)

    rows = list(csv.DictReader(open(out / "per_case/fold0_cases.csv")))
    held_out = get_kfold_indices(n, n_splits=5, fold_index=0, shuffle=True, seed=0)[1]
    assert [int(r["index"]) for r in rows] == list(held_out)
    assert [r["image_id"] for r in rows] == [ids[i] for i in held_out]
    summary = json.loads((out / "per_case/summary.json").read_text())
    name = "mlffm_i01_f0" if arm == "diffusion" else "mlffm_nodiff_f0"
    tested = json.loads((out / f"metrics/{name}_metrics.json").read_text())
    dice = tested.get("metrics/Dice", tested.get("Dice"))
    assert summary["overall"]["pooled_dice"] == pytest.approx(dice, abs=1e-6)
    assert summary["folds"] == [0] and not summary["complete"]
    preds = np.load(out / "per_case/fold0_preds.npz")
    assert list(preds["index"]) == list(held_out) and preds["pred"].shape == (len(held_out), 128, 128)

