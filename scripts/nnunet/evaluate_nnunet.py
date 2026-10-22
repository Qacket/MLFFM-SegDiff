"""Per-case evaluation of nnU-Net test predictions, in the format of scripts/evaluate_per_case.py.

Evaluated cases are the fold0_split == test rows of the split manifest (splits/<dataset>_5fold.csv),
or with --split_column dev_split the dev_val rows (fold 0's test rows are then not read), or with
--all_test every row (a model trained on another dataset, exported with export_nnunet.py --all_test).
Ground truth is sample `index` of <data_dir>/np/Y_tr_<s>x<s>.npy binarized at > 0.5, exactly as for
the other methods; the prediction <pred_dir>/<image_id>.png (nnUNetv2_predict output) is resized to
the ground-truth size with nearest-neighbour interpolation if it differs. With --raw_dataset_dir the
prediction of each index is named as in that export's split_manifest.csv instead (data exported
before the ISIC2018/PH2 id correction: same test indices, other file names). Metrics come from
src/per_case_metrics.case_metrics. Writes fold0_cases.csv, fold0_meta.json, fold0_preds.npz,
oof_cases.csv and summary.json to --out_dir (plus nnUNetPlans.json with --plans_file).
With --paper_kfold (data exported with export_nnunet.py --paper_kfold, trained with
`nnUNetv2_train <ID> 2d <k>` for k = 0..4) every fold k is evaluated on its held-out images from the
validation predictions nnU-Net writes after training with checkpoint_final,
<results_dir>/fold_<k>/validation/<image_id>.png, and fold<k>_* files are written for each fold.
Run in the repository environment (not the nnU-Net one).
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "nnunet"))

from evaluate_per_case import summarize, write_csv  # noqa: E402
from export_nnunet import DATA_DIRS, DATASETS, SPLIT_COLUMNS, paper_folds  # noqa: E402
from per_case_metrics import case_metrics  # noqa: E402


def read_prediction(path, shape):
    """Binary prediction at `shape`; returns (mask, resized)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path)
    resized = image.size != (shape[1], shape[0])
    if resized:
        image = image.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(image) > 0, resized


def evaluate_cases(cases, fold, pred_dir, file_names, y, bf1_tolerance):
    rows, preds, n_resized = [], [], 0
    for case in cases:
        gt = np.asarray(y[int(case["index"]), 0]) > 0.5
        pred, resized = read_prediction(pred_dir / f"{file_names[int(case['index'])]}.png", gt.shape)
        n_resized += resized
        rows.append({"fold": fold, "index": int(case["index"]), "image_id": case["image_id"],
                     **case_metrics(pred, gt, bf1_tolerance=bf1_tolerance)})
        preds.append(pred.astype(np.uint8))
    return rows, preds, n_resized


def write_fold(out_dir, fold, rows, preds, meta):
    write_csv(out_dir / f"fold{fold}_cases.csv", rows)
    np.savez_compressed(out_dir / f"fold{fold}_preds.npz", index=np.array([r["index"] for r in rows]),
                        pred=np.stack(preds))
    (out_dir / f"fold{fold}_meta.json").write_text(json.dumps(meta, indent=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--pred_dir", help="output folder of nnUNetv2_predict")
    parser.add_argument("--paper_kfold", action="store_true",
                        help="evaluate the 5 folds of an export_nnunet.py --paper_kfold dataset (needs --results_dir)")
    parser.add_argument("--results_dir", help="--paper_kfold: nnUNet_results/Dataset<ID>_<NAME>/<trainer>__<plans>__2d")
    parser.add_argument("--trainer", default="nnUNetTrainer", help="nnU-Net trainer class, recorded in the settings")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--nnunet_version", required=True, help="e.g. from `pip show nnunetv2`")
    parser.add_argument("--plans_file", help="nnUNet_preprocessed/Dataset*/nnUNetPlans.json, copied for the record")
    parser.add_argument("--raw_dataset_dir", help="nnUNet_raw/Dataset<ID>_<NAME>; prediction file names come from "
                                                  "its split_manifest.csv (by index)")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--split_file", help="default: splits/<dataset>_5fold.csv")
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--bf1_tolerance", type=float, default=2.0, help="Boundary F1 tolerance in pixels")
    parser.add_argument("--split_column", default="fold0_split", choices=sorted(SPLIT_COLUMNS))
    parser.add_argument("--all_test", action="store_true", help="evaluate every image of the manifest")
    args = parser.parse_args()
    if args.all_test and args.split_column != "fold0_split":
        parser.error("--all_test evaluates all images; do not combine it with --split_column")
    if args.paper_kfold:
        if not args.results_dir or args.pred_dir or args.all_test or args.split_column != "fold0_split":
            parser.error("--paper_kfold needs --results_dir and no --pred_dir, --all_test or --split_column")
    elif not args.pred_dir:
        parser.error("--pred_dir is required")
    eval_value = SPLIT_COLUMNS[args.split_column][1]

    data_dir = Path(args.data_root) / DATA_DIRS.get(args.dataset, args.dataset)
    split_file = Path(args.split_file or ROOT / "splits" / f"{args.dataset}_5fold.csv")
    out_dir = Path(args.out_dir)
    s = args.input_size
    y = np.load(data_dir / "np" / f"Y_tr_{s}x{s}.npy", mmap_mode="r")
    with open(split_file, newline="") as f:
        cases = [r for r in csv.DictReader(f)
                 if args.paper_kfold or args.all_test or r[args.split_column] == eval_value]

    file_names = {int(c["index"]): c["image_id"] for c in cases}
    if args.raw_dataset_dir:
        with open(Path(args.raw_dataset_dir) / "split_manifest.csv", newline="") as f:
            exported = {int(r["index"]): r["image_id"] for r in csv.DictReader(f)
                        if args.paper_kfold or r["subset"] == eval_value}
        if exported.keys() != file_names.keys():
            raise ValueError(f"the test indices of {args.raw_dataset_dir} differ from {split_file}")
        file_names = exported

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plans_file:
        shutil.copy(args.plans_file, out_dir / "nnUNetPlans.json")
    method = (f"nnU-Net v2, 2d configuration, default plans, {args.trainer}, "
              + (f"{args.n_splits}-fold split of the manuscript (splits_final.json), checkpoint_final, "
                 "validation predictions of each fold" if args.paper_kfold else "fold all, checkpoint_final"))
    settings = {"method": method, "nnunet_version": args.nnunet_version,
                "inputs": f"trained and predicted on the {s}x{s} npy cache exported by scripts/nnunet/export_nnunet.py",
                "split_file": str(split_file), "ground_truth": f"np/Y_tr_{s}x{s}.npy > 0.5",
                "prediction_names_from": str(Path(args.raw_dataset_dir) / "split_manifest.csv") if args.raw_dataset_dir
                else "split_file image_id",
                "bf1_tolerance_px": args.bf1_tolerance, "resolution": s}
    if args.all_test:
        settings["evaluated"] = "all images"
    elif args.split_column != "fold0_split":
        settings["evaluated"] = f"{args.split_column} == {eval_value}"

    if args.paper_kfold:
        by_index = {int(c["index"]): c for c in cases}
        if sorted(by_index) != list(range(len(y))):
            raise ValueError(f"{split_file} does not list every cached sample")
        n_resized = 0
        for fold, held_out in enumerate(paper_folds(len(y), args.n_splits)):
            fold_dir = Path(args.results_dir) / f"fold_{fold}"
            rows, preds, resized = evaluate_cases([by_index[i] for i in held_out], fold, fold_dir / "validation",
                                                  file_names, y, args.bf1_tolerance)
            n_resized += resized
            write_fold(out_dir, fold, rows, preds, {"checkpoint": str(fold_dir / "checkpoint_final.pth"),
                                                    **settings, "predictions_resized_nearest": resized})
            print(f"fold {fold}: {len(rows)} cases, mean Dice {np.mean([r['dice'] for r in rows]):.4f}")
        settings["predictions_resized_nearest"] = n_resized
        summary = summarize(out_dir, args.n_splits, settings)
        print(f"saved {out_dir / 'oof_cases.csv'} and summary.json (complete={summary['complete']})")
        return

    rows, preds, n_resized = evaluate_cases(cases, 0, Path(args.pred_dir), file_names, y, args.bf1_tolerance)
    settings["predictions_resized_nearest"] = n_resized
    write_fold(out_dir, 0, rows, preds, {"checkpoint": str(args.pred_dir), **settings})
    summary = summarize(out_dir, args.n_splits, settings)
    print(f"{len(rows)} {eval_value} cases ({n_resized} predictions resized), mean Dice {np.mean([r['dice'] for r in rows]):.4f}; "
          f"saved {out_dir / 'oof_cases.csv'} and summary.json (complete={summary['complete']})")


if __name__ == "__main__":
    main()
