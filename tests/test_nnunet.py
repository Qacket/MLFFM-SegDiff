import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.kfold import get_dev_split, get_kfold_indices  # noqa: E402

N, SIZE = 20, 32
COLUMNS = ("fold,index,image_id,dice,jaccard,precision,recall,specificity,accuracy,hd95,assd,bf1,"
           "boundary_precision,boundary_recall,gt_area,pred_area,tp,fp,fn,empty_pred,empty_gt").split(",")


def make_isic(data_root):
    """Synthetic ISIC2018 folder (image files + npy cache) and a split manifest for fold 0."""
    data_dir = data_root / "ISIC2018"
    ids = [f"ISIC_{i:07d}" for i in range(N)]
    (data_dir / "ISIC2018_Task1-2_Training_Input").mkdir(parents=True)
    for image_id in ids:
        (data_dir / "ISIC2018_Task1-2_Training_Input" / f"{image_id}.jpg").touch()
    (data_dir / "np").mkdir()
    rng = np.random.default_rng(0)
    x = rng.random((N, 3, SIZE, SIZE), dtype=np.float32)
    y = np.zeros((N, 1, SIZE, SIZE), np.float32)
    y[:, :, 8:24, 6:20] = 1
    np.save(data_dir / f"np/X_tr_{SIZE}x{SIZE}.npy", x)
    np.save(data_dir / f"np/Y_tr_{SIZE}x{SIZE}.npy", y)
    train_idx, test_idx = get_kfold_indices(N, n_splits=5, fold_index=0, shuffle=True, seed=0)
    test = set(test_idx.tolist())
    dev_val = set(get_dev_split(train_idx, 0.125, 0)[1].tolist())
    dev = {i: "test" if i in test else "dev_val" if i in dev_val else "dev_train" for i in range(N)}
    split = data_root / "isic2018_5fold.csv"
    with open(split, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_id", "heldout_fold", "fold0_split", "dev_split"])
        writer.writerows([i, ids[i], 0 if i in test else 1, "test" if i in test else "train", dev[i]] for i in range(N))
    return x, y, ids, sorted(test), split


def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return result.stdout


def export_args(tmp_path, split):
    return [ROOT / "scripts/nnunet/export_nnunet.py", "--dataset", "isic2018", "--dataset_id", 501,
            "--raw_dir", tmp_path / "raw", "--data_root", tmp_path, "--split_file", split, "--input_size", SIZE]


def test_export_and_evaluate(tmp_path):
    x, y, ids, test, split = make_isic(tmp_path)
    run(*export_args(tmp_path, split))
    dataset = tmp_path / "raw/Dataset501_ISIC2018"

    exported = list(csv.DictReader(open(dataset / "split_manifest.csv")))
    assert sorted(int(r["index"]) for r in exported if r["subset"] == "test") == test
    assert sorted(p.name for p in (dataset / "imagesTs").iterdir()) == sorted(f"{ids[i]}_0000.png" for i in test)
    meta = json.loads((dataset / "dataset.json").read_text())
    assert meta["numTraining"] == N - len(test) == len(list((dataset / "imagesTr").iterdir()))
    assert meta["channel_names"] == {"0": "R", "1": "G", "2": "B"} and meta["file_ending"] == ".png"
    i = test[0]
    image = np.asarray(Image.open(dataset / f"imagesTs/{ids[i]}_0000.png"))
    assert image.shape == (SIZE, SIZE, 3)
    assert np.array_equal(image, np.round(x[i].transpose(1, 2, 0) * 255).astype(np.uint8))
    assert np.array_equal(np.asarray(Image.open(dataset / f"labelsTs/{ids[i]}.png")), y[i, 0].astype(np.uint8))

    # predictions: the first test case empty, the second at twice the size (resized back), the rest perfect
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    for k, i in enumerate(test):
        label = (y[i, 0] > 0.5).astype(np.uint8)
        if k == 0:
            label = label * 0
        image = Image.fromarray(label)
        if k == 1:
            image = image.resize((2 * SIZE, 2 * SIZE), Image.NEAREST)
        image.save(pred_dir / f"{ids[i]}.png")
    plans = tmp_path / "nnUNetPlans.json"
    plans.write_text('{"plans_name": "nnUNetPlans"}')
    out = tmp_path / "per_case"
    run(ROOT / "scripts/nnunet/evaluate_nnunet.py", "--dataset", "isic2018", "--pred_dir", pred_dir, "--out_dir", out,
        "--nnunet_version", "test", "--plans_file", plans, "--data_root", tmp_path, "--split_file", split,
        "--input_size", SIZE)
    with open(out / "oof_cases.csv", newline="") as f:
        reader = csv.DictReader(f)
        cases = list(reader)
    assert reader.fieldnames == COLUMNS
    assert [int(r["index"]) for r in cases] == test and {r["fold"] for r in cases} == {"0"}
    assert float(cases[0]["dice"]) == 0 and float(cases[0]["hd95"]) == pytest.approx(np.hypot(SIZE, SIZE))
    assert all(float(r["dice"]) == 1 for r in cases[1:])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["settings"]["predictions_resized_nearest"] == 1
    assert summary["overall"]["n_cases"] == len(test)
    assert json.loads((out / "nnUNetPlans.json").read_text())["plans_name"] == "nnUNetPlans"

    # the output pairs with another per_case directory in compare_per_case.py
    ref = tmp_path / "ref"
    ref.mkdir()
    shutil.copy(out / "oof_cases.csv", ref / "oof_cases.csv")
    run(ROOT / "scripts/compare_per_case.py", "--ref", f"ref={ref}", "--compare", f"nnUNet={out}", "--folds", "0",
        "--bootstrap", 100, "--out_dir", tmp_path / "cmp")
    assert (tmp_path / "cmp/comparison.csv").exists()

    with pytest.raises(AssertionError):  # an existing dataset folder is not overwritten
        run(*export_args(tmp_path, split))


def test_export_rejects_mismatched_split(tmp_path):
    _, _, ids, _, split = make_isic(tmp_path)
    rows = split.read_text().splitlines()
    rows[1], rows[2] = rows[1].replace(ids[0], ids[1]), rows[2].replace(ids[1], ids[0])  # swap two image ids
    split.write_text("\n".join(rows) + "\n")
    result = subprocess.run([sys.executable, *map(str, export_args(tmp_path, split))], capture_output=True, text=True)
    assert result.returncode != 0 and "does not match" in result.stderr
    assert not (tmp_path / "raw/Dataset501_ISIC2018").exists()


def test_evaluate_after_id_correction(tmp_path):
    """Exported under old ids; the corrected manifest keeps every index's split but renames the images."""
    _, y, ids, test, split = make_isic(tmp_path)
    run(*export_args(tmp_path, split))
    dataset = tmp_path / "raw/Dataset501_ISIC2018"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    for i in test:
        Image.fromarray((y[i, 0] > 0.5).astype(np.uint8)).save(pred_dir / f"{ids[i]}.png")
    rows = list(csv.DictReader(open(split)))
    corrected = tmp_path / "corrected.csv"
    with open(corrected, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows({**r, "image_id": f"NEW_{int(r['index']):03d}"} for r in rows)
    args = [ROOT / "scripts/nnunet/evaluate_nnunet.py", "--dataset", "isic2018", "--pred_dir", pred_dir,
            "--out_dir", tmp_path / "out", "--nnunet_version", "test", "--data_root", tmp_path,
            "--split_file", corrected, "--input_size", SIZE]
    run(*args, "--raw_dataset_dir", dataset)
    cases = list(csv.DictReader(open(tmp_path / "out/oof_cases.csv")))
    assert [r["image_id"] for r in cases] == [f"NEW_{i:03d}" for i in test]
    assert all(float(r["dice"]) == 1 for r in cases)

    with pytest.raises(AssertionError):  # without the export's names the predictions are not found
        run(*args)
    moved = [{**r, "fold0_split": "test" if int(r["index"]) == 0 else "train"} for r in rows]
    with open(corrected, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(moved)
    with pytest.raises(AssertionError):  # a different test set is refused
        run(*args, "--raw_dataset_dir", dataset)


def test_measure_network_and_sliding_window():
    sys.path.insert(0, str(ROOT / "scripts/nnunet"))
    import torch
    from measure_nnunet import measure_network, sliding_window_positions

    conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
    r = measure_network(conv, (3, 16, 16), torch.device("cpu"), forwards_per_image=4, batch_size=2,
                        warmup=1, repeats=2)
    assert r["parameters"] == 8 * 3 * 3 * 3
    assert r["forward_gflops"] * 1e9 == pytest.approx(2 * 8 * 3 * 9 * 16 * 16)
    assert r["inference_gflops_per_image"] == pytest.approx(4 * r["forward_gflops"])
    assert r["latency_seconds_per_image_batch1"] == pytest.approx(4 * r["forward_latency_seconds_batch1"])
    assert r["peak_memory_gb_batch"] is None
    assert sliding_window_positions((128, 128), (128, 128)) == 1
    assert sliding_window_positions((100, 100), (128, 128)) == 1
    assert sliding_window_positions((256, 256), (128, 128)) == 9  # steps of half a patch: 3 per axis


def test_dev_split_export_and_evaluate(tmp_path):
    """dev_train -> imagesTr, dev_val -> imagesTs; fold 0's test rows are neither exported nor read."""
    _, y, ids, test, split = make_isic(tmp_path)
    rows = list(csv.DictReader(open(split)))
    dev_train = [int(r["index"]) for r in rows if r["dev_split"] == "dev_train"]
    dev_val = [int(r["index"]) for r in rows if r["dev_split"] == "dev_val"]
    assert dev_val and not set(dev_val) & set(test)
    run(ROOT / "scripts/nnunet/export_nnunet.py", "--dataset", "isic2018", "--dataset_id", 511, "--raw_dir",
        tmp_path / "raw", "--data_root", tmp_path, "--split_file", split, "--input_size", SIZE, "--split_column", "dev_split")
    dataset = tmp_path / "raw/Dataset511_ISIC2018DEV"
    assert sorted(p.name for p in (dataset / "imagesTr").iterdir()) == sorted(f"{ids[i]}_0000.png" for i in dev_train)
    assert sorted(p.name for p in (dataset / "imagesTs").iterdir()) == sorted(f"{ids[i]}_0000.png" for i in dev_val)
    exported = list(csv.DictReader(open(dataset / "split_manifest.csv")))
    assert {r["subset"] for r in exported} == {"dev_train", "dev_val"}
    assert not {int(r["index"]) for r in exported} & set(test)
    assert json.loads((dataset / "dataset.json").read_text())["numTraining"] == len(dev_train)

    pred_dir = tmp_path / "pred"  # predictions only for dev_val: reading any test case would fail
    pred_dir.mkdir()
    for i in dev_val:
        Image.fromarray((y[i, 0] > 0.5).astype(np.uint8)).save(pred_dir / f"{ids[i]}.png")
    out = tmp_path / "out"
    run(ROOT / "scripts/nnunet/evaluate_nnunet.py", "--dataset", "isic2018", "--pred_dir", pred_dir, "--out_dir", out,
        "--nnunet_version", "test", "--data_root", tmp_path, "--split_file", split, "--input_size", SIZE,
        "--split_column", "dev_split", "--raw_dataset_dir", dataset)
    with open(out / "oof_cases.csv", newline="") as f:
        reader = csv.DictReader(f)
        cases = list(reader)
    assert reader.fieldnames == COLUMNS
    assert [int(r["index"]) for r in cases] == dev_val and all(float(r["dice"]) == 1 for r in cases)
    assert json.loads((out / "summary.json").read_text())["settings"]["evaluated"] == "dev_split == dev_val"


def test_all_test_export_and_evaluate(tmp_path):
    """--all_test: every image is a test image (testing a model trained on another dataset)."""
    _, y, ids, _, split = make_isic(tmp_path)
    run(ROOT / "scripts/nnunet/export_nnunet.py", "--dataset", "isic2018", "--dataset_id", 512, "--raw_dir",
        tmp_path / "raw", "--data_root", tmp_path, "--split_file", split, "--input_size", SIZE, "--all_test")
    dataset = tmp_path / "raw/Dataset512_ISIC2018ALL"
    assert not list((dataset / "imagesTr").iterdir())
    assert sorted(p.name for p in (dataset / "imagesTs").iterdir()) == sorted(f"{i}_0000.png" for i in ids)
    assert {r["subset"] for r in csv.DictReader(open(dataset / "split_manifest.csv"))} == {"test"}
    assert json.loads((dataset / "dataset.json").read_text())["numTraining"] == 0

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    for i in range(N):
        Image.fromarray((y[i, 0] > 0.5).astype(np.uint8)).save(pred_dir / f"{ids[i]}.png")
    out = tmp_path / "out"
    args = [ROOT / "scripts/nnunet/evaluate_nnunet.py", "--dataset", "isic2018", "--pred_dir", pred_dir, "--out_dir", out,
            "--nnunet_version", "test", "--data_root", tmp_path, "--split_file", split, "--input_size", SIZE,
            "--raw_dataset_dir", dataset, "--all_test"]
    run(*args)
    with open(out / "oof_cases.csv", newline="") as f:
        reader = csv.DictReader(f)
        cases = list(reader)
    assert reader.fieldnames == COLUMNS
    assert [int(r["index"]) for r in cases] == list(range(N)) and {r["fold"] for r in cases} == {"0"}
    assert all(float(r["dice"]) == 1 for r in cases)
    assert json.loads((out / "summary.json").read_text())["settings"]["evaluated"] == "all images"
    with pytest.raises(AssertionError):  # --all_test does not combine with --split_column
        run(*args, "--split_column", "dev_split")


def test_paper_kfold_export_and_evaluate(tmp_path):
    """--paper_kfold: every image in imagesTr, the manuscript's 5 folds in splits_final.json, and per-fold
    evaluation of nnU-Net's validation predictions."""
    _, y, ids, _, _ = make_isic(tmp_path)
    folds = [get_kfold_indices(N, n_splits=5, fold_index=k, shuffle=True, seed=0)[1].tolist() for k in range(5)]
    heldout = {i: k for k, fold in enumerate(folds) for i in fold}
    split = tmp_path / "kfold.csv"
    with open(split, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_id", "heldout_fold", "fold0_split"])
        writer.writerows([i, ids[i], heldout[i], "test" if heldout[i] == 0 else "train"] for i in range(N))
    def export(split_file, dataset_id):
        return [ROOT / "scripts/nnunet/export_nnunet.py", "--dataset", "isic2018", "--dataset_id", dataset_id,
                "--raw_dir", tmp_path / "raw", "--data_root", tmp_path, "--split_file", split_file,
                "--input_size", SIZE, "--paper_kfold"]
    run(*export(split, 511))
    dataset = tmp_path / "raw/Dataset511_ISIC2018KF"
    assert sorted(p.name for p in (dataset / "imagesTr").iterdir()) == sorted(f"{i}_0000.png" for i in ids)
    assert not (dataset / "imagesTs").exists()
    splits = json.loads((dataset / "splits_final.json").read_text())
    assert [s["val"] for s in splits] == [[ids[i] for i in fold] for fold in folds]
    assert all(sorted(s["train"] + s["val"]) == sorted(ids) for s in splits)
    manifest = list(csv.DictReader(open(dataset / "split_manifest.csv")))
    assert [r["subset"] for r in manifest] == [f"fold{heldout[i]}" for i in range(N)]

    # an ungrouped manifest whose folds differ from the paper split is rejected
    bad = tmp_path / "bad.csv"
    rows = list(csv.reader(open(split)))
    rows[1][2] = str((int(rows[1][2]) + 1) % 5)
    csv.writer(open(bad, "w", newline="")).writerows(rows)
    with pytest.raises(AssertionError, match="differs from the paper split"):
        run(*export(bad, 513))

    # validation predictions: perfect, except an empty one for the first image of fold 2
    results = tmp_path / "results"
    for k, fold in enumerate(folds):
        (results / f"fold_{k}/validation").mkdir(parents=True)
        for n, i in enumerate(fold):
            label = (y[i, 0] > 0.5).astype(np.uint8) * (0 if (k, n) == (2, 0) else 1)
            Image.fromarray(label).save(results / f"fold_{k}/validation/{ids[i]}.png")
    out = tmp_path / "out"
    evaluate = [ROOT / "scripts/nnunet/evaluate_nnunet.py", "--dataset", "isic2018", "--out_dir", out,
                "--nnunet_version", "test", "--data_root", tmp_path, "--split_file", split, "--input_size", SIZE]
    run(*evaluate, "--paper_kfold", "--results_dir", results)
    with open(out / "oof_cases.csv", newline="") as f:
        reader = csv.DictReader(f)
        cases = list(reader)
    assert reader.fieldnames == COLUMNS
    assert sorted(int(r["index"]) for r in cases) == list(range(N))
    assert all(int(r["fold"]) == heldout[int(r["index"])] for r in cases)
    assert [int(r["index"]) for r in cases if float(r["dice"]) == 0] == [folds[2][0]]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] and summary["folds"] == [0, 1, 2, 3, 4]
    assert json.loads((out / "fold3_meta.json").read_text())["checkpoint"].endswith("fold_3/checkpoint_final.pth")
    with pytest.raises(AssertionError):  # --paper_kfold needs --results_dir
        run(*evaluate, "--paper_kfold")
