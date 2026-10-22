import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_per_case as cp  # noqa: E402


def cases(dice, hd95, ids=None):
    return {(0, i): {"fold": "0", "index": str(i), "image_id": ids[i] if ids else f"ISIC_{i}",
                     "dice": str(d), "hd95": str(h)} for i, (d, h) in enumerate(zip(dice, hd95))}


def test_holm_and_rank_biserial():
    assert np.allclose(cp.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert cp.rank_biserial(np.array([0.1, 0.2, 0.3])) == 1.0
    assert cp.rank_biserial(np.array([0.1, -0.1, 0.0])) == 0.0


def test_compare_detects_shift_and_direction():
    rng = np.random.default_rng(0)
    dice = rng.uniform(0.6, 0.95, 300)
    hd = rng.uniform(2, 20, 300)
    ref = ("ref", cases(dice, hd))
    better = ("better", cases(dice + 0.02 + rng.normal(0, 0.01, 300), hd - 2 + rng.normal(0, 0.5, 300)))
    same = ("same", cases(dice, hd))
    res = {(r["method"], r["metric"]): r for r in cp.compare(ref, [better, same], ["dice", "hd95"], n_boot=2000)}
    b = res[("better", "dice")]
    assert b["mean_diff_ci_low"] < 0.02 < b["mean_diff_ci_high"] and b["mean_diff_ci_low"] > 0
    assert b["holm_p"] < 1e-6 and b["rank_biserial"] > 0.9 and b["favours"] == "better"
    assert res[("better", "hd95")]["favours"] == "better"  # lower HD95 favours the method
    s = res[("same", "dice")]
    assert s["wilcoxon_p"] == 1.0 and s["favours"] == "none" and s["mean_diff_ci_low"] == s["mean_diff_ci_high"] == 0


def test_mismatched_cases_and_nan():
    ref = ("ref", cases([0.9, 0.8, 0.7], [1, 2, 3]))
    with pytest.raises(ValueError, match="different cases"):
        cp.compare(ref, [("m", cases([0.9, 0.8], [1, 2]))], ["dice"], n_boot=10)
    res = cp.compare(ref, [("m", cases([0.9, float("nan"), 0.75], [1, 2, 3]))], ["dice"], n_boot=10)
    assert res[0]["n"] == 2


def test_cluster_bootstrap_widens_ci_for_duplicated_cases():
    rng = np.random.default_rng(1)
    d = rng.normal(0.02, 0.05, 150)
    dice_ref = np.repeat(rng.uniform(0.6, 0.9, 150), 2)  # each lesion imaged twice, identical results
    dice_m = dice_ref + np.repeat(d, 2)
    ids = [f"ISIC_{i}" for i in range(300)]
    clusters = {f"ISIC_{i}": f"L{i // 2}" for i in range(300)}
    args = (("ref", cases(dice_ref, dice_ref, ids)), [("m", cases(dice_m, dice_m, ids))], ["dice"], 4000)
    plain = cp.compare(*args)[0]
    grouped = cp.compare(*args, cluster_of=clusters)[0]
    width = lambda r: r["mean_diff_ci_high"] - r["mean_diff_ci_low"]
    assert width(grouped) > 1.25 * width(plain)


def test_cli(tmp_path):
    rng = np.random.default_rng(2)
    dice = rng.uniform(0.6, 0.95, 50)
    for name, shift in [("ref", 0.0), ("a1", -0.03), ("a7", 0.001)]:
        d = tmp_path / name / "per_case"
        d.mkdir(parents=True)
        with open(d / "oof_cases.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold", "index", "image_id", "dice", "hd95", "bf1"])
            w.writeheader()
            for i, v in enumerate(dice):
                w.writerow({"fold": 0, "index": i, "image_id": f"ISIC_{i}", "dice": v + shift + rng.normal(0, 0.005),
                            "hd95": 5, "bf1": 0.8})
    res = cp.main(["--ref", f"E4={tmp_path / 'ref/per_case'}", "--compare", f"A1={tmp_path / 'a1/per_case'}",
                   f"A7={tmp_path / 'a7/per_case'}", "--metrics", "dice", "hd95", "--bootstrap", "500",
                   "--out_dir", str(tmp_path / "out")])
    assert {r["method"] for r in res} == {"A1", "A7"}
    rows = list(csv.DictReader(open(tmp_path / "out/comparison.csv")))
    assert len(rows) == 4 and next(r for r in rows if r["method"] == "A1" and r["metric"] == "dice")["favours"] == "E4"
    assert "Holm correction" in (tmp_path / "out/comparison.md").read_text()
