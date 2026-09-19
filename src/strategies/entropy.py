"""Uncertainty sampling driven by the most uncertain frame of each segment."""

from __future__ import annotations

from .backends import to_numpy
from .base import QueryStrategy


class Entropy(QueryStrategy):
    """Select the segments with the highest binary entropy at their most uncertain frame and class."""

    required_properties = ("prediction_soft",)

    def _select(self, ctx):
        xp = ctx.xp
        probabilities = ctx.get("prediction_soft")[ctx.unlabeled]  # [n_unlabeled, F, C]
        safe = xp.clip(probabilities, 1e-12, 1.0 - 1e-12)
        binary_entropy = -(safe * xp.log(safe) + (1.0 - safe) * xp.log(1.0 - safe))
        scores = binary_entropy.max(axis=-1).max(axis=-1)  # [n_unlabeled]
        positions = xp.argpartition(scores, int(scores.shape[0]) - ctx.step_size)[-ctx.step_size:]
        return scores[positions], ctx.unlabeled[to_numpy(positions)]
