"""Disagreement between the trained model and a nearest-neighbour classifier fitted on the labeled frames."""

from __future__ import annotations

import numpy as np

from .auxiliary import nearest_labeled_targets
from .backends import to_backend, to_numpy
from .base import QueryStrategy


def mismatch_scores(ctx) -> np.ndarray:
    """Score every unlabeled segment by the frame where the model and the 1-NN reference disagree most.

    Args:
        ctx: Query context with the ``prediction_hard`` property prepared.

    Returns:
        float32 scores aligned with ``ctx.unlabeled``, ``[n_unlabeled]``. A score is the largest per-frame
        fraction of classes on which the two hard predictions differ.
    """
    xp = ctx.xp
    labeled_features, labeled_targets = ctx.labeled_frames()
    reference = nearest_labeled_targets(
        labeled_features, labeled_targets, ctx.frame_embedding, ctx.unlabeled, xp
    )
    model_prediction = ctx.get("prediction_hard")[ctx.unlabeled]
    frame_scores = (to_backend(reference, xp) != model_prediction).mean(axis=-1, dtype=xp.float32)  # [n_unlabeled, F]
    return to_numpy(frame_scores.max(axis=-1)).astype(np.float32)


def split_at_threshold(scores: np.ndarray, n_select: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Split candidate positions into those strictly above the cut-off score and those tied at it.

    Args:
        scores: Scores aligned with the candidate pool.
        n_select: Number of candidates to select, smaller than the pool size.

    Returns:
        ``(above_positions, tied_positions, threshold)``, where threshold is the n_select-th largest score.
    """
    threshold = float(np.partition(scores, -n_select)[-n_select])
    return np.flatnonzero(scores > threshold), np.flatnonzero(scores == threshold), threshold


class Disagreement(QueryStrategy):
    """Select the segments where the model disagrees most with a 1-NN classifier fitted on the labeled frames."""

    required_properties = ("prediction_hard",)

    def _select(self, ctx):
        scores = mismatch_scores(ctx)
        if ctx.step_size >= scores.size:
            return scores, ctx.unlabeled
        above, tied, _ = split_at_threshold(scores, ctx.step_size)
        # The cut-off is the step_size-th largest score, so the tied group always fills at least one slot.
        remaining = ctx.step_size - above.size
        # Break the tie at the threshold by preferring the smaller global index.
        tied = tied[np.argpartition(ctx.unlabeled[tied], remaining - 1)[:remaining]]
        positions = np.concatenate([above, tied])
        return scores[positions], ctx.unlabeled[positions]
