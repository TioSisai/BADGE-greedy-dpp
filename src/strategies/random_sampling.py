"""Random sampling baseline based on skactiveml RandomSampling."""

from __future__ import annotations

import numpy as np
from skactiveml.pool import RandomSampling as SkactivemlRandomSampling

from .base import QueryStrategy


class RandomSampling(QueryStrategy):
    """Uniform random selection through skactiveml, also used for the shared cold start."""

    def _select(self, ctx):
        if ctx.step_size == ctx.unlabeled.size:
            # Taking every candidate needs no draw.
            return np.ones(ctx.step_size, dtype=np.float32), ctx.unlabeled
        # skactiveml seeds each draw from a copy of random_state and the missing-label count of y, so rounds differ.
        y = np.full(ctx.n_train, np.nan, dtype=float)
        y[ctx.labeled] = 0.0
        sampler = SkactivemlRandomSampling(missing_label=np.nan, random_state=self.random_state)
        selected = sampler.query(
            np.zeros((ctx.n_train, 1), dtype=np.float32),
            y,
            candidates=ctx.unlabeled,
            batch_size=ctx.step_size,
        )
        selected = np.asarray(selected, dtype=np.int64)
        return np.ones(selected.shape, dtype=np.float32), selected
