"""Determinantal point process traversals over BADGE gradient embeddings.

``greedy_dpp`` is the proposed selector and ``mcmc_dpp`` is the swap sampler of the original BADGE repository.
"""

from __future__ import annotations

import heapq

import numpy as np
from opt_einsum import contract

from .backends import is_cupy, solve_lower_triangular

# Ridge term keeping the empty-set Gram matrix invertible, matching the paper.
LAMBDA_REG = 1e-6


def _empty_cholesky(num_features: int, xp):
    """Return the Cholesky factor of ``LAMBDA_REG * I``, the kernel of an empty selection."""
    return np.sqrt(LAMBDA_REG) * xp.eye(num_features, dtype=xp.float64)


def _marginal_gains(current_l, candidates, xp):
    """Return ``1 + phi^T K^-1 phi`` for every row of candidates, with current_l the Cholesky factor of K."""
    solved = solve_lower_triangular(current_l, candidates.T, xp)
    return contract("ij,ij->j", solved, solved) + 1.0


def _extend_cholesky(current_l, row, xp):
    """Add ``row^T row`` to the kernel by a rank-one update of its factor, which keeps the factor D by D."""
    return xp.linalg.qr(xp.vstack((current_l.T, row[None, :])), mode="r").T


def greedy_dpp(X, n: int, xp):
    """Select n candidates by greedy maximization of the regularized log-volume ``log det(lambda I + Phi_S^T Phi_S)``.

    A marginal gain never grows while the selection grows, so on numpy the gains are refreshed lazily from
    a max-heap of stale values. On cupy every step rescores the whole pool in one batched solve.

    Args:
        X: Candidate gradient embeddings, float64 ``[N, D]``. Every row is eligible.
        n: Number of candidates to select.
        xp: Array module the inputs live in.

    Returns:
        ``(scores, positions)``, where a score is the marginal volume gain ``1 + phi^T K^-1 phi``
        at the moment of selection.
    """
    num_candidates, num_features = X.shape
    select_scores = xp.empty(n, dtype=xp.float64)
    select_positions = xp.empty(n, dtype=xp.int64)
    current_l = _empty_cholesky(num_features, xp)

    if not is_cupy(xp):
        heap =[(-gain, pos) for pos, gain in enumerate(_marginal_gains(current_l, X, xp).tolist())]
        heapq.heapify(heap)
        for i in range(n):
            while True:
                _, best_pos = heapq.heappop(heap)
                best_gain = float(_marginal_gains(current_l, X[best_pos:best_pos + 1], xp)[0])
                # Comparing (gain, position) pairs breaks ties like argmax does, by the smallest position.
                if not heap or (-best_gain, best_pos) <= heap[0]:
                    break
                heapq.heappush(heap, (-best_gain, best_pos))
            select_scores[i] = best_gain
            select_positions[i] = best_pos
            current_l = _extend_cholesky(current_l, X[best_pos], xp)
        return select_scores, select_positions

    available_mask = xp.ones(num_candidates, dtype=bool)
    for i in range(n):
        masked_gains = xp.where(available_mask, _marginal_gains(current_l, X, xp), -xp.inf)
        best_pos = masked_gains.argmax()
        select_scores[i] = masked_gains[best_pos]
        select_positions[i] = best_pos
        available_mask[best_pos] = False
        current_l = _extend_cholesky(current_l, X[best_pos], xp)
    return select_scores, select_positions


def _gram_red(current_kernel, current_kernel_inv, keep_positions, remove_pos):
    """Drop the row and column at remove_pos from the kernel and downdate its inverse."""
    kept_inv_rows = current_kernel_inv[keep_positions]
    reduced_kernel = current_kernel[keep_positions][:, keep_positions]
    column = kept_inv_rows[:, remove_pos].reshape(-1, 1)
    diagonal = current_kernel_inv[remove_pos, remove_pos].reshape(1, 1)
    return reduced_kernel, kept_inv_rows[:, keep_positions] - (column @ column.T) / diagonal


def _gram_aug(reduced_kernel, reduced_kernel_inv, b_vec, c_val, xp):
    """Append one row and column to the kernel and update its inverse, returning the Schur complement gain."""
    gain = c_val - b_vec.T @ reduced_kernel_inv @ b_vec
    helper = reduced_kernel_inv @ b_vec
    augmented_kernel = xp.concatenate(
        (
            xp.concatenate((reduced_kernel, b_vec), axis=1),
            xp.concatenate((b_vec.T, c_val), axis=1),
        ),
        axis=0,
    )
    augmented_kernel_inv = xp.concatenate(
        (
            xp.concatenate((reduced_kernel_inv + (helper @ helper.T) / gain, -helper / gain), axis=1),
            xp.concatenate(((-helper / gain).T, 1.0 / gain), axis=1),
        ),
        axis=0,
    )
    return augmented_kernel, augmented_kernel_inv, gain


def _removal_gain(reduced_features, target_feature, reduced_kernel_inv):
    """Return the Schur complement of removing one member, i.e. its marginal contribution to the determinant."""
    coupling = reduced_features @ target_feature.T
    self_kernel = target_feature @ target_feature.T
    return self_kernel - coupling.T @ reduced_kernel_inv @ coupling


def mcmc_dpp(X, n: int, xp):
    """Sample a fixed-size subset by the swap-based MCMC chain of the original BADGE implementation.

    The chain targets ``det(Phi_S Phi_S^T)`` and runs ``int(5 n log n) - 1`` swap proposals, each accepted with
    probability ``min(1, added_gain / removed_gain)``.

    Args:
        X: Candidate gradient embeddings, float64 ``[N, D]``. Every row is eligible.
        n: Size of the sampled subset.
        xp: Array module the inputs live in.

    Returns:
        ``(scores, positions)``. The original implementation returns no scores, so scores are all ones.
    """
    num_candidates = X.shape[0]
    subset_positions = xp.random.choice(num_candidates, size=n, replace=False).astype(xp.int64, copy=False)
    in_subset = xp.zeros(num_candidates, dtype=bool)
    in_subset[subset_positions] = True
    current_kernel = X[subset_positions] @ X[subset_positions].T
    current_kernel_inv = xp.linalg.pinv(current_kernel)
    base_positions = xp.arange(n - 1)

    for _ in range(1, int(5 * n * np.log(n))):
        outside_positions = xp.flatnonzero(~in_subset)
        remove_local_pos = xp.random.randint(0, n)
        remove_candidate_pos = xp.asarray(subset_positions[remove_local_pos], dtype=xp.int64).reshape(-1)
        add_candidate_pos = xp.asarray(
            outside_positions[xp.random.randint(0, outside_positions.shape[0])], dtype=xp.int64
        ).reshape(-1)

        # Integer positions skip the removed member, since a boolean mask would cost cupy a host sync per use.
        keep_positions = base_positions + (base_positions >= remove_local_pos)
        kept_positions = subset_positions[keep_positions]
        reduced_kernel, reduced_kernel_inv = _gram_red(
            current_kernel, current_kernel_inv, keep_positions, remove_local_pos
        )
        reduced_features = X[kept_positions]
        removed_gain = _removal_gain(reduced_features, X[remove_candidate_pos], reduced_kernel_inv)
        added_feature = X[add_candidate_pos]
        augmented_kernel, augmented_kernel_inv, added_gain = _gram_aug(
            reduced_kernel,
            reduced_kernel_inv,
            reduced_features @ added_feature.T,
            added_feature @ added_feature.T,
            xp,
        )

        accept = xp.asarray(
            xp.random.random() <= xp.minimum(1.0, added_gain / removed_gain).reshape(()), dtype=bool
        )
        updated_in_subset = in_subset.copy()
        updated_in_subset[remove_candidate_pos] = False
        updated_in_subset[add_candidate_pos] = True
        in_subset = xp.where(accept, updated_in_subset, in_subset)
        subset_positions = xp.where(
            accept, xp.concatenate((kept_positions, add_candidate_pos)), subset_positions
        )
        current_kernel = xp.where(accept, augmented_kernel, current_kernel)
        current_kernel_inv = xp.where(accept, augmented_kernel_inv, current_kernel_inv)

    return xp.ones(n, dtype=xp.float64), subset_positions
