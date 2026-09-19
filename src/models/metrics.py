"""Frame-wise multilabel mAP and normalized area under the learning curve."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torchmetrics.functional.classification import multilabel_precision_recall_curve


def precision_recall_curves(frame_proba: torch.Tensor, frame_targets: torch.Tensor):
    """Compute the per-class precision-recall curves that both average precision and the F1 thresholds read.

    Args:
        frame_proba: Probabilities of all frames, ``[M, C]``.
        frame_targets: int64 0/1 targets of the same frames on the same device, ``[M, C]``.

    Returns:
        ``(precision, recall, thresholds)``, each a list of C tensors on the device of the inputs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No positive samples found in target")
        # The targets come from the cache as exact 0/1, so the input validation pass is skipped.
        return multilabel_precision_recall_curve(
            frame_proba, frame_targets, num_labels=frame_targets.shape[1], validate_args=False
        )


def average_precision_tensors(curves) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce precision-recall curves to average precision, the step-wise area under each curve, on the device.

    Args:
        curves: Return value of precision_recall_curves.

    Returns:
        ``(per_class, macro)`` as float32 tensors ``[C]`` and ``[]``.
    """
    precision, recall, _ = curves
    per_class = torch.stack([-torch.sum((r[1:] - r[:-1]) * p[:-1]) for p, r in zip(precision, recall)])
    return per_class, per_class.mean()


def mean_average_precision(curves) -> tuple[np.ndarray, float]:
    """Return per-class average precision and its macro average on the host, in a single transfer.

    Args:
        curves: Return value of precision_recall_curves.

    Returns:
        ``(per_class, macro)``, with per-class average precision as float64 ``[C]``.
    """
    per_class, macro = average_precision_tensors(curves)
    packed = torch.cat([per_class, macro.reshape(1)]).cpu().numpy().astype(np.float64)
    return packed[:-1], float(packed[-1])


def frame_wise_map(frame_proba: torch.Tensor, frame_targets: torch.Tensor) -> tuple[np.ndarray, float]:
    """Compute per-class average precision over all frames and its macro average.

    Args:
        frame_proba: Probabilities of all frames, ``[M, C]``.
        frame_targets: int64 0/1 targets of the same frames on the same device, ``[M, C]``.

    Returns:
        ``(per_class, macro)``, with per-class average precision as float64 ``[C]``.
    """
    return mean_average_precision(precision_recall_curves(frame_proba, frame_targets))


def cumulative_aulc(xs, ys) -> float:
    """Compute normalized AULC by dividing the trapezoidal integral by the x-axis span.

    Args:
        xs: Cumulative labeled counts increasing with each round.
        ys: mAP for the corresponding rounds.

    Returns:
        Normalized area, or NaN if there are fewer than two points.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.shape[0] < 2:
        return float("nan")
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))
