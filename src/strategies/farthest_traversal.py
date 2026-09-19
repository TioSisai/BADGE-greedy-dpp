"""Coreset-style farthest-first traversal over segment representations."""

from __future__ import annotations

from .base import QueryStrategy
from .traversal import farthest_traversal


class FarthestTraversal(QueryStrategy):
    """Select the segments farthest from the labeled set in the pooled embedding space."""

    required_properties = ("seg_embedding",)

    def _select(self, ctx):
        return farthest_traversal(ctx.get("seg_embedding"), ctx.step_size, ctx.labeled, ctx.unlabeled, ctx.xp)
