"""The paper configs use the paper's split and relative paths; the split manifests match results/."""
import csv
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.kfold import get_kfold_indices  # noqa: E402

CONFIGS = sorted((ROOT / "configs/paper").rglob("*.yaml"))
SIZES = {"isic2018": 2594, "ph2": 200, "ham10000": 10015}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def values(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from values(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def test_all_configs_present():
    names = {str(p.relative_to(ROOT / "configs/paper")) for p in CONFIGS}
    expected = {f"{ds}/{m}.yaml" for ds in SIZES for m in ("mlffm", "dermosegdiff", "unet", "swinunetr", "generic_unet")}
    expected |= {"ham10000/dermosegdiff_bs64.yaml", "ph2/diffusion_vs_nondiffusion/diffusion.yaml",
                 "ph2/diffusion_vs_nondiffusion/nondiffusion.yaml"}
    assert names == expected


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT / "configs/paper")))
def test_config(path):
    sys.path.insert(0, str(ROOT / "src"))
    from models import MODELS as DIFFUSION_MODELS
    from seg_models import MODELS as SEG_MODELS
    cfg = yaml.safe_load(path.read_text())
    kfold = cfg["run"]["kfold"]
    assert (kfold["n_splits"], kfold["shuffle"], kfold["seed"]) == (5, True, 0)
    assert not {"group_by", "dev_fraction", "inner_fold", "folds", "all_test"} & set(kfold)
    assert "prior" not in cfg["dataset"] and not cfg.get("testing", {}).get("clip_x0", False)
    assert cfg["dataset"]["data_dir"].rstrip("/") == "data/" + {"isic2018": "ISIC2018"}.get(cfg["dataset"]["name"],
                                                                                         cfg["dataset"]["name"])
    for key, value in values(cfg):
        if key.endswith("dir") and isinstance(value, str):
            assert not value.startswith("/"), (key, value)
    models = SEG_MODELS if cfg["run"].get("task") == "seg" else DIFFUSION_MODELS
    assert cfg["model"]["class"] in models


@pytest.mark.parametrize("dataset", sorted(SIZES))
def test_split_manifest_is_the_paper_split(dataset):
    rows = read(ROOT / "splits" / f"{dataset}_5fold.csv")
    n = SIZES[dataset]
    assert [int(r["index"]) for r in rows] == list(range(n))
    for k in range(5):
        held_out = get_kfold_indices(n, 5, k, shuffle=True, seed=0)[1]
        assert {int(r["index"]) for r in rows if int(r["heldout_fold"]) == k} == set(held_out.tolist())
    assert all(r["fold0_split"] == ("test" if r["heldout_fold"] == "0" else "train") for r in rows)
    folds = {r["image_id"]: int(r["fold"]) for r in read(ROOT / "results/folds" / f"{dataset}_folds.csv")}
    assert {r["image_id"]: int(r["heldout_fold"]) for r in rows} == folds
    for path in (ROOT / "results/per_image" / dataset).glob("*.csv"):
        assert {(int(r["index"]), r["image_id"]) for r in read(path)} == {(int(r["index"]), r["image_id"]) for r in rows}
