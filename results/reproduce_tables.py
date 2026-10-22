"""Recompute the values of Tables 4-8 and the diffusion vs. non-diffusion comparison from the files in results/.

Usage (from the repository root):
    python results/reproduce_tables.py [--ham_metadata path/to/HAM10000_metadata.csv]

Requires numpy and scipy. Table 7 uses scripts/compare_per_case.py of this repository. For HAM10000,
the paper resamples all images of a lesion together in the bootstrap; pass the HAM10000 metadata file
(image_id, lesion_id; available from the HAM10000 Dataverse page) to reproduce these intervals exactly.
Without it, the HAM10000 intervals are computed by resampling images and differ slightly.
"""
import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

DATASETS = {"isic2018": "ISIC2018", "ph2": "PH2", "ham10000": "HAM10000"}
BASELINES = {"dermosegdiff": "DermoSegDiff", "generic_unet": "Generic U-Net", "swinunetr": "SwinUNETR",
             "unet": "U-Net"}
ALL = {**BASELINES, "nnunet": "nnU-Net", "mlffm_segdiff": "MLFFM-SegDiff"}
METRICS = ["accuracy", "f1", "jaccard", "precision", "recall", "specificity", "dice"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_sd(values):
    return f"{st.mean(values):.4f} ± {st.pstdev(values):.4f}"


def holm(p):
    order = sorted(range(len(p)), key=p.__getitem__)
    adjusted, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def paired_t(a, b):
    """Mean difference a - b over folds with 95% t-interval, p-value, and Cohen's d_z."""
    d = np.array(a) - np.array(b)
    m, s = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, len(d) - 1) * s / np.sqrt(len(d))
    return m, m - half, m + half, float(stats.ttest_rel(a, b).pvalue), m / s, int((d > 0).sum())


def per_image_fold_values(ds, model):
    """Per-fold pooled Dice, mean per-image Dice, mean HD95, and mean ASSD (Table 6)."""
    by_fold = defaultdict(list)
    for r in read(HERE / "per_image" / ds / f"{model}.csv"):
        by_fold[int(r["fold"])].append(r)
    out = defaultdict(list)
    for k in sorted(by_fold):
        rows = by_fold[k]
        tp, fp, fn = (sum(int(r[c]) for r in rows) for c in ("tp", "fp", "fn"))
        out["pooled_dice"].append(2 * tp / (2 * tp + fp + fn))
        for c in ("dice", "hd95", "assd"):
            out[c].append(st.mean(float(r[c]) for r in rows))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ham_metadata", help="HAM10000 metadata CSV with image_id and lesion_id columns")
    args = parser.parse_args()

    # Table 4 and Table 5
    main_rows = read(HERE / "fold_level/main_comparison.csv")
    table4 = defaultdict(lambda: defaultdict(list))
    for r in main_rows:
        for m in METRICS:
            table4[r["dataset"], r["model"]][m].append(float(r[m]))
    print("Table 4 (mean ± population SD over five folds):", " | ".join(METRICS))
    for ds, name in DATASETS.items():
        for model in [*BASELINES, "mlffm_segdiff"]:
            print(f"  {name:9s} {ALL[model]:14s}", " | ".join(mean_sd(table4[ds, model][m]) for m in METRICS))
    print("\nTable 5 (unweighted mean of the three dataset means):")
    for model in [*BASELINES, "mlffm_segdiff"]:
        print(f"  {ALL[model]:14s}", " | ".join(f"{st.mean(st.mean(table4[ds, model][m]) for ds in DATASETS):.4f}"
                                              for m in METRICS))

    # Table 6
    print("\nTable 6: pooled Dice | per-image Dice | HD95 | ASSD")
    reeval = {(ds, model): per_image_fold_values(ds, model) for ds in DATASETS for model in ALL}
    for ds, name in DATASETS.items():
        for model in [*BASELINES, "nnunet", "mlffm_segdiff"]:
            v = reeval[ds, model]
            cells = [mean_sd(v["pooled_dice"]), mean_sd(v["dice"])]
            cells += [f"{st.mean(v[c]):.2f} ± {st.pstdev(v[c]):.2f}" for c in ("hd95", "assd")]
            print(f"  {name:9s} {ALL[model]:14s}", " | ".join(cells))

    # Table 7
    from compare_per_case import compare, load_cases, read_clusters
    clusters = read_clusters(args.ham_metadata, "image_id", "lesion_id") if args.ham_metadata else None
    print("\nTable 7 (positive = favours MLFFM-SegDiff): Dice mean [95% CI], median, Holm p, r | "
          "HD95 mean [95% CI], median, Holm p, r")
    for ds, name in DATASETS.items():
        ref = ("MLFFM-SegDiff", load_cases(HERE / "per_image" / ds / "mlffm_segdiff.csv"))
        methods = [(ALL[m], load_cases(HERE / "per_image" / ds / f"{m}.csv")) for m in [*BASELINES, "nnunet"]]
        res = compare(ref, methods, ["dice", "hd95"], 10000, 0, clusters if ds == "ham10000" else None)
        res = {(r["method"], r["metric"]): r for r in res}
        for method, _ in methods:
            cells = []
            for metric, flip, nd in (("dice", -1, 4), ("hd95", 1, 2)):
                r = res[method, metric]
                lo, hi = sorted(flip * r[k] for k in ("mean_diff_ci_low", "mean_diff_ci_high"))
                cells.append(f"{flip * r['mean_diff']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}] "
                             f"{flip * r['median_diff']:+.{nd}f} p={r['holm_p']:.3g} r={flip * r['rank_biserial']:+.2f}")
            print(f"  {name:9s} {method:14s}", " | ".join(cells))

    # Table 8
    print("\nTable 8A (Dice of Table 4; paired t-test over folds, Holm across the four methods):")
    for ds, name in DATASETS.items():
        res = [paired_t(table4[ds, "mlffm_segdiff"]["dice"], table4[ds, b]["dice"]) for b in BASELINES]
        for b, (m, lo, hi, p, dz, n), ph in zip(BASELINES, res, holm([r[3] for r in res])):
            print(f"  {name:9s} {ALL[b]:14s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f} p_Holm={ph:.3f} "
                  f"d_z={dz:+.2f} folds={n}/5")
    print("Table 8B (re-evaluation, MLFFM-SegDiff vs nnU-Net):")
    for ds, name in DATASETS.items():
        a, b = reeval[ds, "mlffm_segdiff"], reeval[ds, "nnunet"]
        m, lo, hi, p, dz, n = paired_t(a["pooled_dice"], b["pooled_dice"])
        print(f"  {name:9s} pooled Dice {m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f} d_z={dz:+.2f} folds={n}/5")
        m, lo, hi, p, dz, n = paired_t(b["hd95"], a["hd95"])
        print(f"  {name:9s} HD95 (px)   {m:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.3f} d_z={dz:+.2f} folds={n}/5")

    # Diffusion vs. non-diffusion on PH2
    print("\nPH2, same protocol (1,000 epochs, batch size 12, no EMA, final epoch): Dice")
    by_variant = defaultdict(list)
    for r in read(HERE / "fold_level/diffusion_vs_nondiffusion_ph2.csv"):
        by_variant[r["variant"]].append(float(r["dice"]))
    for variant, values in by_variant.items():
        print(f"  {variant:13s} {mean_sd(values)}")


if __name__ == "__main__":
    main()
