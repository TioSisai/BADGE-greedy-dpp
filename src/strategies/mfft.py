"""Mismatch-first farthest traversal, which diversifies the group of segments tied at the mismatch cut-off."""

from __future__ import annotations

import numpy as np

from .backends import to_numpy
from .base import QueryStrategy
from .disagreement import mismatch_scores, split_at_threshold
from .traversal import farthest_traversal


class MFFT(QueryStrategy):
    """Take the segments above the mismatch cut-off, then fill the batch by farthest traversal within the tied group."""

    required_properties = ("seg_embedding", "prediction_hard")

    def _select(self, ctx):
        scores = mismatch_scores(ctx)
        if ctx.step_size >= scores.size:
            return scores, ctx.unlabeled
        above, tied, threshold = split_at_threshold(scores, ctx.step_size)
        above_idxes = ctx.unlabeled[above]
        # The cut-off is the step_size-th largest score, so the tied group always fills at least one slot.
        remaining = ctx.step_size - above_idxes.size

        # Segments already accepted this round join the labeled set as coverage anchors.
        anchors = np.concatenate([ctx.labeled, above_idxes])
        _, tied_idxes = farthest_traversal(ctx.get("seg_embedding"), remaining, anchors, ctx.unlabeled[tied], ctx.xp)
        tied_idxes = to_numpy(tied_idxes).astype(np.int64)
        tied_scores = np.full(tied_idxes.shape, threshold, dtype=scores.dtype)
        return np.concatenate([scores[above], tied_scores]), np.concatenate([above_idxes, tied_idxes])
