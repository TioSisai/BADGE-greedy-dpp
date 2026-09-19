"""k-means++ seeding over BADGE gradient embeddings, following scikit-learn's local-trial rule."""

from __future__ import annotations

import numpy as np
from opt_einsum import contract

from .backends import is_cupy, to_numpy


def _squared_distances(X, Y, x_squared_norms, xp):
    """Return squared Euclidean distances from each row of Y to every row of X, ``[n_y, n_x]``."""
    y_squared_norms = contract("ij,ij->i", Y, Y)
    distances = y_squared_norms[:, None] + x_squared_norms[None, :] - 2 * contract("id,jd->ij", Y, X)
    xp.maximum(distances, 0, out=distances)
    return distances


def kmeans_plusplus(X, n: int, xp, *, random_state=None):
    """Pick n seeds by scikit-learn's k-means++ rule with uniform sample weights.

    Every step draws ``2 + int(log(n))`` local trials and keeps the one that lowers the potential most.

    Args:
        X: Candidate representations, ``[N, D]``. Every row is eligible.
        n: Number of seeds to select.
        xp: Array module the inputs live in.
        random_state: Seed or numpy random state driving the local trials.

    Returns:
        ``(scores, positions)``, where a score is the squared distance to the closest earlier seed and
        the first seed's score is NaN because no earlier seed exists.
    """
    n_samples = X.shape[0]
    rng = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    n_local_trials = 2 + int(np.log(n))

    x_squared_norms = contract("ij,ij->i", X, X)
    select_positions = xp.empty(n, dtype=xp.int64)
    select_scores = xp.empty(n, dtype=X.dtype)

    # The explicit uniform p mirrors scikit-learn's weighted choice, which draws differently from choice without p.
    first_pos = int(rng.choice(n_samples, p=np.full(n_samples, 1.0 / n_samples)))
    select_positions[0] = first_pos
    closest_dist_sq = _squared_distances(X, X[first_pos:first_pos + 1], x_squared_norms, xp).reshape(-1)
    current_pot = float(to_numpy(closest_dist_sq.sum()))

    for step in range(1, n):
        rand_vals = rng.uniform(size=n_local_trials) * current_pot
        if is_cupy(xp):
            rand_vals = xp.asarray(rand_vals, dtype=X.dtype)
        candidate_positions = xp.searchsorted(xp.cumsum(closest_dist_sq), rand_vals, side="left")
        xp.clip(candidate_positions, 0, n_samples - 1, out=candidate_positions)

        distance_to_candidates = _squared_distances(X, X[candidate_positions], x_squared_norms, xp)
        xp.minimum(distance_to_candidates, closest_dist_sq[None, :], out=distance_to_candidates)
        candidates_pot = to_numpy(distance_to_candidates.sum(axis=1))
        best_trial_pos = int(candidates_pot.argmin())
        current_pot = float(candidates_pot[best_trial_pos])
        best_pos = candidate_positions[best_trial_pos]
        select_scores[step] = closest_dist_sq[best_pos]
        select_positions[step] = best_pos
        closest_dist_sq = distance_to_candidates[best_trial_pos]

    select_scores[0] = xp.nan
    return select_scores, select_positions
