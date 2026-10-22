import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.ham10000 import PrepareHAM10000  # noqa: E402
from datasets.isic2018 import PrepareISIC2018  # noqa: E402



def make_isic(root, order):
    """Masks on disk are 256x256 rectangles; the cache stores them (at 128) in `order`."""
    (root / "ISIC2018_Task1-2_Training_Input").mkdir(parents=True)
    (root / "ISIC2018_Task1_Training_GroundTruth").mkdir()
    (root / "np").mkdir()
    ids = [f"ISIC_{i:07d}" for i in range(12)]
    cached = {}
    for k, image_id in enumerate(ids):
        (root / "ISIC2018_Task1-2_Training_Input" / f"{image_id}.jpg").touch()
        m = np.zeros((256, 256), np.uint8)
        m[10 + 8 * k:120 + 8 * k, 20:60 + 10 * k] = 255
        Image.fromarray(m).save(root / "ISIC2018_Task1_Training_GroundTruth" / f"{image_id}_segmentation.png")
        cached[image_id] = (m[::2, ::2] > 0).astype(np.float32)[None]
    np.save(root / "np/Y_tr_128x128.npy", np.stack([cached[ids[i]] for i in order]))
    return ids


def test_get_ids_prefers_the_id_file(tmp_path):
    ids = make_isic(tmp_path, list(range(12)))
    prep = PrepareISIC2018(data_dir=str(tmp_path), image_size=128)
    assert prep.get_ids() == ids  # no id file: sorted file names
    (tmp_path / "np/ids_128x128.txt").write_text("\n".join(reversed(ids)) + "\n")
    assert prep.get_ids() == list(reversed(ids))


def test_building_a_cache_writes_its_order(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    ids = [f"ISIC_{i:07d}" for i in (5, 2, 9)]
    for k, image_id in enumerate(ids):
        Image.fromarray(np.full((16, 16, 3), 40 * k, np.uint8)).save(tmp_path / "images" / f"{image_id}.jpg")
        Image.fromarray(np.full((16, 16), 255 * (k % 2), np.uint8)).save(tmp_path / "masks" / f"{image_id}_segmentation.png")
    PrepareHAM10000(data_dir=str(tmp_path), image_size=8).get_data()
    assert (tmp_path / "np/ids_8x8.txt").read_text().split() == sorted(ids)


def test_refresh_ids_in_per_case_results(tmp_path):
    import evaluate_per_case as epc
    ids = make_isic(tmp_path / "data", list(range(12)))
    (tmp_path / "data/np/ids_128x128.txt").write_text("\n".join(reversed(ids)) + "\n")
    cfg = yaml.safe_load((ROOT / "configs/paper/isic2018/unet.yaml").read_text())
    cfg["dataset"]["data_dir"] = str(tmp_path / "data")
    cfg["run"]["kfold"] = {"metrics_dir": str(tmp_path / "metrics")}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "per_case"
    out.mkdir()
    rows = [{c: 0 for c in epc.COLUMNS} | {"fold": 0, "index": i, "image_id": "wrong", "tp": 5, "empty_pred": False, "empty_gt": False}
            for i in (3, 7)]
    epc.write_csv(out / "fold0_cases.csv", rows)
    (out / "fold0_meta.json").write_text(json.dumps({"checkpoint": "x"}))
    sys.argv = ["evaluate_per_case", "-c", str(path), "--summary_only", "--refresh_ids"]
    epc.main()
    got = [r["image_id"] for r in csv.DictReader(open(out / "fold0_cases.csv"))]
    assert got == [list(reversed(ids))[3], list(reversed(ids))[7]]
