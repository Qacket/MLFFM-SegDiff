"""Per-case segmentation metrics on binary 2-D masks (evaluated at the model resolution).

Conventions (reported with the results):
- Boundary: mask minus its erosion by a 3x3 cross; the image edge counts as boundary (medpy).
- HD95: 95th percentile of the pooled surface distances in both directions (medpy), in pixels.
- ASSD: mean of the pooled surface distances in both directions (medpy assd), in pixels.
- Boundary F1: boundary precision/recall with a distance tolerance in pixels.
- Empty masks: both empty -> Dice/Jaccard/BF1 = 1 and HD95 = ASSD = 0; exactly one empty ->
  Dice/Jaccard/BF1 = 0 and HD95 = ASSD = image diagonal. Precision is undefined (NaN) for an empty
  prediction, recall for an empty ground truth.
"""
import numpy as np
from scipy import ndimage

_CROSS = ndimage.generate_binary_structure(2, 1)


def boundary(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=_CROSS)


def _surface_distances(pred, gt):
    """Distances from each boundary pixel of one mask to the boundary of the other."""
    pb, gb = boundary(pred), boundary(gt)
    to_gt = ndimage.distance_transform_edt(~gb)[pb]
    to_pred = ndimage.distance_transform_edt(~pb)[gb]
    return to_gt, to_pred


def _ratio(num, den):
    return num / den if den else float("nan")


def case_metrics(pred, gt, bf1_tolerance=2.0):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError(f"expected two 2-D masks of the same shape, got {pred.shape} and {gt.shape}")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    empty_pred, empty_gt = not pred.any(), not gt.any()
    out = {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "accuracy": (tp + tn) / pred.size,
        "gt_area": tp + fn,
        "pred_area": tp + fp,
        "tp": tp, "fp": fp, "fn": fn,
        "empty_pred": empty_pred,
        "empty_gt": empty_gt,
    }
    if empty_pred and empty_gt:
        out.update(dice=1.0, jaccard=1.0, hd95=0.0, assd=0.0, bf1=1.0, boundary_precision=1.0,
                   boundary_recall=1.0)
    elif empty_pred or empty_gt:
        diagonal = float(np.hypot(*pred.shape))
        out.update(dice=0.0, jaccard=0.0, hd95=diagonal, assd=diagonal, bf1=0.0,
                   boundary_precision=0.0, boundary_recall=0.0)
    else:
        to_gt, to_pred = _surface_distances(pred, gt)
        distances = np.concatenate([to_gt, to_pred])
        bp = float(np.mean(to_gt <= bf1_tolerance))
        br = float(np.mean(to_pred <= bf1_tolerance))
        out.update(
            dice=2 * tp / (2 * tp + fp + fn),
            jaccard=tp / (tp + fp + fn),
            hd95=float(np.percentile(distances, 95)),
            assd=float(np.mean(distances)),
            bf1=2 * bp * br / (bp + br) if bp + br else 0.0,
            boundary_precision=bp,
            boundary_recall=br,
        )
    return out
