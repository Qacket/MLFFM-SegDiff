"""Paired per-case comparison of segmentation results (output of evaluate_per_case.py).

Each method is given as NAME=PATH, where PATH is a per_case directory (its oof_cases.csv is used)
or a cases CSV. Cases are paired on (fold, index); the case sets must match. For every compared
method and metric (difference = method - reference):
- n pairs (cases where either value is NaN are dropped for that metric);
- mean and median of the paired differences with 95% percentile bootstrap CIs, resampling cases,
  or whole clusters (e.g. lesions) when --cluster_file is given;
- two-sided Wilcoxon signed-rank test (zero differences dropped), Holm-corrected across the
  compared methods within each metric;
- matched-pairs rank-biserial correlation (positive = method values larger).
For metrics where lower is better (hd95, assd) a negative difference favours the method.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

LOWER_IS_BETTER = {"hd95", "assd"}
DEFAULT_METRICS = ["dice", "jaccard", "hd95", "bf1", "precision", "recall"]
COLUMNS = ["method", "reference", "metric", "n", "reference_mean", "method_mean",
           "mean_diff", "mean_diff_ci_low", "mean_diff_ci_high",
           "median_diff", "median_diff_ci_low", "median_diff_ci_high",
           "wilcoxon_p", "holm_p", "rank_biserial", "favours"]


def load_cases(path, folds=None):
    path = Path(path)
    if path.is_dir():
        path = path / "oof_cases.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if folds is not None:
        rows = [r for r in rows if int(r["fold"]) in folds]
    cases = {(int(r["fold"]), int(r["index"])): r for r in rows}
    if len(cases) != len(rows):
        raise ValueError(f"duplicate (fold, index) rows in {path}")
    return cases


def rank_biserial(d):
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(d, clusters, n_boot, rng):
    """95% percentile CIs of the mean and median of d, resampling clusters (one per case if None)."""
    if clusters is None:  # resample cases, vectorized
        samples = d[rng.integers(0, d.size, (n_boot, d.size))]
        return (np.percentile(samples.mean(1), [2.5, 97.5]),
                np.percentile(np.median(samples, 1), [2.5, 97.5]))
    keys, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(keys.size)]
    means, medians = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, keys.size, keys.size)
        sample = d[np.concatenate([members[k] for k in picked])]
        means[b], medians[b] = sample.mean(), np.median(sample)
    return np.percentile(means, [2.5, 97.5]), np.percentile(medians, [2.5, 97.5])


def compare(reference, methods, metrics, n_boot=10000, seed=0, cluster_of=None):
    """reference / methods: (name, cases); returns one result dict per method and metric."""
    ref_name, ref_cases = reference
    results = []
    for metric in metrics:
        rows = []
        for name, cases in methods:
            if cases.keys() != ref_cases.keys():
                missing, extra = ref_cases.keys() - cases.keys(), cases.keys() - ref_cases.keys()
                raise ValueError(f"{name} and {ref_name} have different cases "
                                 f"({len(missing)} missing, {len(extra)} extra)")
            keys = sorted(ref_cases)
            ref = np.array([float(ref_cases[k][metric]) for k in keys])
            other = np.array([float(cases[k][metric]) for k in keys])
            ok = ~(np.isnan(ref) | np.isnan(other))
            d = other[ok] - ref[ok]
            clusters = None
            if cluster_of is not None:
                ids = [ref_cases[k]["image_id"] for k, keep in zip(keys, ok) if keep]
                clusters = np.array([cluster_of.get(i) or f"__single__{i}" for i in ids])
            mean_ci, median_ci = bootstrap_ci(d, clusters, n_boot, np.random.default_rng(seed))
            p = 1.0 if not np.any(d) else float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
            rows.append({"method": name, "reference": ref_name, "metric": metric, "n": int(ok.sum()),
                         "reference_mean": float(ref[ok].mean()), "method_mean": float(other[ok].mean()),
                         "mean_diff": float(d.mean()), "mean_diff_ci_low": mean_ci[0], "mean_diff_ci_high": mean_ci[1],
                         "median_diff": float(np.median(d)), "median_diff_ci_low": median_ci[0],
                         "median_diff_ci_high": median_ci[1], "wilcoxon_p": p, "rank_biserial": rank_biserial(d)})
        for row, adjusted in zip(rows, holm([r["wilcoxon_p"] for r in rows])):
            row["holm_p"] = float(adjusted)
            sign = -1 if metric in LOWER_IS_BETTER else 1
            row["favours"] = ("none" if adjusted >= 0.05 else
                              row["method"] if sign * row["mean_diff"] > 0 else row["reference"])
        results += rows
    return results


def read_clusters(path, id_column, cluster_column):
    with open(path, newline="") as f:
        return {r[id_column]: r[cluster_column] for r in csv.DictReader(f)}


def markdown(results):
    lines = ["| Method vs reference | Metric | n | Ref. mean | Method mean | Mean diff [95% CI] | "
             "Median diff [95% CI] | Wilcoxon p | Holm p | r_rb | Favours |",
             "|---|---|---:|---:|---:|---|---|---:|---:|---:|---|"]
    for r in results:
        lines.append(
            f"| {r['method']} vs {r['reference']} | {r['metric']} | {r['n']} | {r['reference_mean']:.4f} | "
            f"{r['method_mean']:.4f} | {r['mean_diff']:+.4f} [{r['mean_diff_ci_low']:+.4f}, {r['mean_diff_ci_high']:+.4f}] | "
            f"{r['median_diff']:+.4f} [{r['median_diff_ci_low']:+.4f}, {r['median_diff_ci_high']:+.4f}] | "
            f"{r['wilcoxon_p']:.2e} | {r['holm_p']:.2e} | {r['rank_biserial']:+.3f} | {r['favours']} |")
    return "\n".join(lines) + "\n"


def parse_named(value):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=parse_named, required=True, help="NAME=PATH of the reference method")
    parser.add_argument("--compare", type=parse_named, nargs="+", required=True, help="NAME=PATH ...")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--folds", type=str, help="comma-separated folds to keep (default: all rows)")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster_file", help="CSV mapping image ids to clusters (e.g. HAM10000_metadata)")
    parser.add_argument("--id_column", default="image_id")
    parser.add_argument("--cluster_column", default="lesion_id")
    parser.add_argument("--out_dir", help="also write comparison.csv and comparison.md here")
    args = parser.parse_args(argv)

    folds = {int(f) for f in args.folds.split(",")} if args.folds else None
    reference = (args.ref[0], load_cases(args.ref[1], folds))
    methods = [(name, load_cases(path, folds)) for name, path in args.compare]
    cluster_of = read_clusters(args.cluster_file, args.id_column, args.cluster_column) if args.cluster_file else None
    results = compare(reference, methods, args.metrics, args.bootstrap, args.seed, cluster_of)
    table = markdown(results)
    print(table)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "comparison.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(results)
        (out / "comparison.md").write_text(
            f"Reference: {args.ref[0]}; bootstrap: {args.bootstrap} resamples "
            f"({'clusters: ' + args.cluster_column if cluster_of else 'cases'}); "
            f"Holm correction within each metric.\n\n" + table)
    return results


if __name__ == "__main__":
    main()
