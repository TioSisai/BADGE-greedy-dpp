"""Farthest-first traversal over segment representations, anchored at the already selected set."""

from __future__ import annotations

from .backends import pairwise_distances


def farthest_traversal(X, n: int, anchor_idxes, candidate_idxes, xp):
    """Greedily take the candidate whose nearest anchor is farthest away in Euclidean distance.

    Args:
        X: Segment representations for the full pool, ``[N, D]``.
        n: Number of candidates to select.
        anchor_idxes: Global indices already covered, which must not be empty.
        candidate_idxes: Global indices eligible for selection, disjoint from anchor_idxes.
        xp: Array module the inputs live in.

    Returns:
        ``(scores, idxes)``, where a score is the nearest-anchor distance at the moment of selection.
    """
    anchor_idxes = xp.asarray(anchor_idxes, dtype=xp.int64).reshape(-1)
    candidate_idxes = xp.asarray(candidate_idxes, dtype=xp.int64).reshape(-1)

    select_dists = xp.empty(n, dtype=xp.float64)
    select_idxes = xp.empty(n, dtype=xp.int64)
    available_mask = xp.ones(candidate_idxes.shape[0], dtype=bool)
    candidate_X = X[candidate_idxes]
    min_dists = pairwise_distances(candidate_X, X[anchor_idxes]).min(axis=1)

    for i in range(n):
        masked_min_dists = xp.where(available_mask, min_dists, -xp.inf)
        next_pos = masked_min_dists.argmax()
        select_dists[i] = masked_min_dists[next_pos]
        next_idx = candidate_idxes[next_pos]
        select_idxes[i] = next_idx
        available_mask[next_pos] = False
        new_dists = pairwise_distances(candidate_X, X[next_idx:next_idx + 1]).reshape(-1)
        xp.minimum(min_dists, new_dists, out=min_dists)
    return select_dists, select_idxes
