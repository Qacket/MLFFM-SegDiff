"""Export the fixed 80/20 split (or the development split) to an nnU-Net v2 raw dataset.

The split is a column of the split manifest (splits/<dataset>_5fold.csv): --split_column fold0_split
(default; train / test) or dev_split (dev_train / dev_val inside fold 0's training set; its test rows are skipped, not exported). The
images and masks are the dataset's npy cache at --input_size, i.e. the arrays every other method
trains and tests on. With --all_test every manifest row becomes a test image (nothing is exported
for training; NAME gets an ALL suffix): for testing a model trained on another dataset, e.g. the
ISIC2018 nnU-Net on all PH2 images. With --paper_kfold every image goes to imagesTr and the five
folds of the original manuscript (utils.kfold.get_kfold_indices: shuffle, seed 0, no grouping, as in
configs/paper/<dataset>) are written to splits_final.json, for `nnUNetv2_train <ID> 2d <k>`
(NAME gets a KF suffix; subset in split_manifest.csv = fold<k> holding the image out). Each manifest
row must match the cache order (index, image_id) or the export stops. Written to
<raw_dir>/Dataset<ID>_<NAME>/ (NAME gets a DEV suffix for dev_split):
  imagesTr/<image_id>_0000.png  RGB uint8 (cached image in [0, 1] x 255)   train / dev_train
  labelsTr/<image_id>.png       uint8 0/1 (cached mask > 0.5)
  imagesTs/, labelsTs/          test / dev_val
  dataset.json                  2D natural-image dataset, channels R/G/B, labels background/lesion
  split_manifest.csv            index, image_id, subset of every exported sample
  splits_final.json             --paper_kfold: copy to nnUNet_preprocessed/Dataset<ID>_<NAME>/ before training
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.ham10000 import PrepareHAM10000  # noqa: E402
from datasets.isic2018 import PrepareISIC2018  # noqa: E402
from datasets.ph2 import PreparePH2  # noqa: E402
from utils.kfold import get_kfold_indices  # noqa: E402

DATASETS = ["isic2018", "ham10000", "ph2"]
# dataset directory names under --data_root, where they differ from the dataset key
DATA_DIRS = {"isic2018": "ISIC2018"}
PREPARERS = {"isic2018": PrepareISIC2018, "ph2": PreparePH2, "ham10000": PrepareHAM10000}
NNUNET_NAMES = {"isic2018": "ISIC2018", "ph2": "PH2", "ham10000": "HAM10000"}
# manifest column -> (value exported for training, value exported as the nnU-Net test set);
# rows with any other value (the fold-0 test rows of dev_split) are skipped
SPLIT_COLUMNS = {"fold0_split": ("train", "test"), "dev_split": ("dev_train", "dev_val")}


def read_split(split_file, ids, column="fold0_split"):
    """Manifest rows, checked against the image ids in cache order."""
    with open(split_file, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ids):
        raise ValueError(f"{split_file} has {len(rows)} rows but the data has {len(ids)} samples")
    for position, row in enumerate(rows):
        if int(row["index"]) != position or row["image_id"] != ids[position]:
            raise ValueError(f"{split_file} row {position} ({row['index']}, {row['image_id']}) does not match "
                             f"the data ({position}, {ids[position]})")
        if column not in row:
            raise ValueError(f"{split_file} has no column {column!r}")
        allowed = {"fold0_split": ("train", "test"), "dev_split": ("dev_train", "dev_val", "test")}[column]
        if row[column] not in allowed:
            raise ValueError(f"unexpected {column} {row[column]!r} in row {position}")
    return rows


def save_case(dataset_dir, img_dir, lbl_dir, x, y, index, image_id):
    image = np.round(np.clip(x[index].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    label = (y[index, 0] > 0.5).astype(np.uint8)
    Image.fromarray(image).save(dataset_dir / img_dir / f"{image_id}_0000.png")
    Image.fromarray(label).save(dataset_dir / lbl_dir / f"{image_id}.png")


def paper_folds(n_samples, n_splits=5, seed=0):
    """Held-out indices of each fold of the original manuscript's k-fold split."""
    return [get_kfold_indices(n_samples, n_splits, k, shuffle=True, seed=seed)[1].tolist() for k in range(n_splits)]


def export_paper_kfold(dataset, data_dir, input_size, split_file, dataset_dir, n_splits=5, seed=0):
    prep = PREPARERS[dataset](data_dir=str(data_dir), image_size=input_size)
    data = prep.get_data()
    x, y = data["x"], data["y"]
    ids = prep.get_ids()
    if len(ids) != len(x):
        raise ValueError(f"{len(ids)} image files but {len(x)} cached samples")
    rows = read_split(split_file, ids)  # checks the manifest against the cache order
    folds = paper_folds(len(ids), n_splits, seed)
    heldout = {i: k for k, fold in enumerate(folds) for i in fold}
    if "group" not in rows[0]:  # ungrouped manifests (ISIC2018, PH2) hold the same folds
        for row in rows:
            if int(row["heldout_fold"]) != heldout[int(row["index"])]:
                raise ValueError(f"{split_file} row {row['index']}: heldout_fold {row['heldout_fold']} "
                                 f"differs from the paper split ({heldout[int(row['index'])]})")
    if dataset_dir.exists():
        raise FileExistsError(f"{dataset_dir} exists; remove it or choose another dataset id")
    (dataset_dir / "imagesTr").mkdir(parents=True)
    (dataset_dir / "labelsTr").mkdir(parents=True)
    for index, image_id in enumerate(ids):
        save_case(dataset_dir, "imagesTr", "labelsTr", x, y, index, image_id)
    (dataset_dir / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(ids),
        "file_ending": ".png",
        "description": f"{dataset} {n_splits}-fold split of the original manuscript (shuffle, seed {seed}, "
                       f"splits_final.json), npy cache at {input_size}x{input_size}; exported by "
                       f"scripts/nnunet/export_nnunet.py --paper_kfold",
    }, indent=4))
    (dataset_dir / "splits_final.json").write_text(json.dumps(
        [{"train": [ids[i] for i in range(len(ids)) if heldout[i] != k], "val": [ids[i] for i in fold]}
         for k, fold in enumerate(folds)], indent=1))
    with open(dataset_dir / "split_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image_id", "subset"])
        writer.writeheader()
        writer.writerows({"index": i, "image_id": ids[i], "subset": f"fold{heldout[i]}"} for i in range(len(ids)))
    return len(ids)


def export(dataset, data_dir, input_size, split_file, dataset_dir, column="fold0_split", all_test=False):
    prep = PREPARERS[dataset](data_dir=str(data_dir), image_size=input_size)
    data = prep.get_data()  # builds the npy cache if it does not exist yet
    x, y = data["x"], data["y"]
    ids = prep.get_ids()
    if len(ids) != len(x):
        raise ValueError(f"{len(ids)} image files but {len(x)} cached samples")
    train_value, test_value = SPLIT_COLUMNS[column]
    rows = [r for r in read_split(split_file, ids, column) if r[column] in (train_value, test_value)]
    if all_test:  # every image is a test image
        rows = [{**r, column: test_value} for r in rows]
    if dataset_dir.exists():
        raise FileExistsError(f"{dataset_dir} exists; remove it or choose another dataset id")
    folders = {train_value: ("imagesTr", "labelsTr"), test_value: ("imagesTs", "labelsTs")}
    for img_dir, lbl_dir in folders.values():
        (dataset_dir / img_dir).mkdir(parents=True)
        (dataset_dir / lbl_dir).mkdir(parents=True)
    for row in rows:
        save_case(dataset_dir, *folders[row[column]], x, y, int(row["index"]), row["image_id"])
    n_train = sum(r[column] == train_value for r in rows)
    (dataset_dir / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": n_train,
        "file_ending": ".png",
        "description": (f"{dataset} all {len(rows)} images as test images ({Path(split_file).name}), " if all_test
                        else f"{dataset} fixed 80/20 split (fold0_split of {Path(split_file).name}), " if column == "fold0_split"
                        else f"{dataset} {column} of {Path(split_file).name} ({train_value} / {test_value}), ")
                       + f"npy cache at {input_size}x{input_size}; exported by scripts/nnunet/export_nnunet.py",
    }, indent=4))
    with open(dataset_dir / "split_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image_id", "subset"])
        writer.writeheader()
        writer.writerows({"index": r["index"], "image_id": r["image_id"], "subset": r[column]} for r in rows)
    return n_train, len(rows) - n_train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--dataset_id", type=int, required=True, help="nnU-Net dataset id, e.g. 501")
    parser.add_argument("--raw_dir", required=True, help="nnUNet_raw")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--split_file", help="default: splits/<dataset>_5fold.csv")
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--split_column", default="fold0_split", choices=sorted(SPLIT_COLUMNS))
    parser.add_argument("--all_test", action="store_true", help="export every image as a test image")
    parser.add_argument("--paper_kfold", action="store_true",
                        help="export every image for training with the manuscript's 5 folds in splits_final.json")
    args = parser.parse_args()
    if args.all_test and args.split_column != "fold0_split":
        parser.error("--all_test exports all images; do not combine it with --split_column")
    if args.paper_kfold and (args.all_test or args.split_column != "fold0_split"):
        parser.error("--paper_kfold does not combine with --all_test or --split_column")
    data_dir = Path(args.data_root) / DATA_DIRS.get(args.dataset, args.dataset)
    split_file = args.split_file or ROOT / "splits" / f"{args.dataset}_5fold.csv"
    name = NNUNET_NAMES[args.dataset] + ("ALL" if args.all_test else "KF" if args.paper_kfold
                                         else "DEV" if args.split_column == "dev_split" else "")
    dataset_dir = Path(args.raw_dir) / f"Dataset{args.dataset_id:03d}_{name}"
    if args.paper_kfold:
        n = export_paper_kfold(args.dataset, data_dir, args.input_size, split_file, dataset_dir)
        print(f"exported {dataset_dir}: {n} cases; copy splits_final.json to nnUNet_preprocessed before training")
        return
    n_train, n_test = export(args.dataset, data_dir, args.input_size, split_file, dataset_dir, args.split_column,
                             args.all_test)
    print(f"exported {dataset_dir}: {n_train} training / {n_test} test cases")


if __name__ == "__main__":
    main()
