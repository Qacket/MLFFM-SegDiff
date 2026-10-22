"""Put the npy cache of a dataset in the image order of the paper's split manifest.

The folds are positions in the cached arrays (<data_dir>/np/X_tr_<s>x<s>.npy and Y_tr_<s>x<s>.npy,
split by src/utils/kfold.get_kfold_indices). The dataset preparers build a new cache in sorted file-name
order, whereas the ISIC2018 and PH2 caches of the paper's experiments follow the order of an unsorted
file listing; splits/<dataset>_5fold.csv records that order (index, image_id). This script builds the
cache if it does not exist yet (as training does), then rewrites X, Y and np/ids_<s>x<s>.txt in the
order of the manifest, so that training, testing and the evaluation scripts use the folds of the paper.
The HAM10000 cache is already in sorted order. Run once per dataset, before training:

  python scripts/order_cache.py --dataset isic2018 --data_dir data/isic2018
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.cache_ids import write_cache_ids  # noqa: E402
from datasets.ham10000 import PrepareHAM10000  # noqa: E402
from datasets.isic2018 import PrepareISIC2018  # noqa: E402
from datasets.ph2 import PreparePH2  # noqa: E402

PREPARERS = {"isic2018": PrepareISIC2018, "ph2": PreparePH2, "ham10000": PrepareHAM10000}


def read_order(split_file):
    with open(split_file, newline="") as f:
        rows = list(csv.DictReader(f))
    if [int(r["index"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{split_file}: rows are not in index order")
    return [r["image_id"] for r in rows]


def order_cache(dataset, data_dir, split_file, input_size=128):
    prep = PREPARERS[dataset](data_dir=str(data_dir), image_size=input_size)
    data = prep.get_data()  # builds the cache if it does not exist
    ids, target = prep.get_ids(), read_order(split_file)
    if len(ids) != len(data["x"]):
        raise ValueError(f"{len(ids)} image ids but {len(data['x'])} cached samples")
    if ids == target:
        return f"{data_dir}: cache already in the order of {split_file}"
    if sorted(ids) != sorted(target):
        missing, extra = sorted(set(target) - set(ids)), sorted(set(ids) - set(target))
        raise ValueError(f"the images differ from {split_file}: {len(missing)} missing (e.g. {missing[:3]}), "
                         f"{len(extra)} not in the manifest (e.g. {extra[:3]})")
    position = {image_id: i for i, image_id in enumerate(ids)}
    order = np.array([position[image_id] for image_id in target])
    npy_dir = Path(data_dir) / "np"
    for key, array in (("X", data["x"]), ("Y", data["y"])):
        path = npy_dir / f"{key}_tr_{input_size}x{input_size}.npy"
        tmp = npy_dir / f"{key}_tr_{input_size}x{input_size}.tmp.npy"
        np.save(tmp, array[order])
        os.replace(tmp, path)
    write_cache_ids(str(npy_dir), input_size, target)
    return f"{data_dir}: reordered {len(target)} cached samples to the order of {split_file}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(PREPARERS))
    parser.add_argument("--data_dir", required=True, help="dataset.data_dir of the configs, e.g. data/isic2018")
    parser.add_argument("--split_file", help="default: splits/<dataset>_5fold.csv")
    parser.add_argument("--input_size", type=int, default=128)
    args = parser.parse_args(argv)
    split_file = args.split_file or ROOT / "splits" / f"{args.dataset}_5fold.csv"
    print(order_cache(args.dataset, args.data_dir, split_file, args.input_size))


if __name__ == "__main__":
    main()
