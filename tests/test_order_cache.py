import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.isic2018 import PrepareISIC2018  # noqa: E402
from order_cache import order_cache  # noqa: E402


def make_cache(root, n=8, size=16):
    """ISIC2018 folder with a cache in sorted file-name order, as the dataset preparer builds it."""
    ids = [f"ISIC_{i:07d}" for i in range(n)]
    (root / "ISIC2018_Task1-2_Training_Input").mkdir(parents=True)
    for image_id in ids:
        (root / "ISIC2018_Task1-2_Training_Input" / f"{image_id}.jpg").touch()
    (root / "np").mkdir()
    x = np.arange(n, dtype=np.float32)[:, None, None, None] * np.ones((1, 3, size, size), np.float32)
    np.save(root / f"np/X_tr_{size}x{size}.npy", x)
    np.save(root / f"np/Y_tr_{size}x{size}.npy", x[:, :1] / n)
    return ids


def write_manifest(path, ids):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_id", "heldout_fold", "fold0_split"])
        writer.writerows([i, image_id, 0, "test"] for i, image_id in enumerate(ids))


def test_reorders_cache_and_ids(tmp_path):
    ids = make_cache(tmp_path / "data")
    order = [3, 0, 7, 1, 6, 2, 5, 4]
    write_manifest(tmp_path / "split.csv", [ids[i] for i in order])
    message = order_cache("isic2018", tmp_path / "data", tmp_path / "split.csv", input_size=16)
    assert "reordered 8" in message
    prep = PrepareISIC2018(data_dir=str(tmp_path / "data"), image_size=16)
    assert prep.get_ids() == [ids[i] for i in order]
    data = prep.get_data()
    assert data["x"][:, 0, 0, 0].tolist() == order
    assert np.allclose(data["y"][:, 0, 0, 0] * 8, order)
    assert not list((tmp_path / "data/np").glob("*.tmp.npy"))
    # a second run finds the cache in order
    assert "already" in order_cache("isic2018", tmp_path / "data", tmp_path / "split.csv", input_size=16)


def test_rejects_other_images(tmp_path):
    ids = make_cache(tmp_path / "data")
    write_manifest(tmp_path / "split.csv", ids[:-1] + ["ISIC_9999999"])
    with pytest.raises(ValueError, match="differ"):
        order_cache("isic2018", tmp_path / "data", tmp_path / "split.csv", input_size=16)
    assert PrepareISIC2018(data_dir=str(tmp_path / "data"), image_size=16).get_ids() == ids  # unchanged
