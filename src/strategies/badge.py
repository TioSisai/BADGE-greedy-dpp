"""BADGE gradient embeddings of the pseudo-labeled loss, traversed by k-means++, MCMC DPP, or greedy DPP."""

from __future__ import annotations

from opt_einsum import contract

from .backends import to_backend, to_numpy
from .base import QueryStrategy
from .dpp import greedy_dpp, mcmc_dpp
from .kmeans_plusplus import kmeans_plusplus


def badge_gradient(proba, last_feature, xp):
    """Build the BADGE gradient embedding of every segment.

    The gradient of the frame-wise pseudo-labeled binary cross entropy with respect to the output layer is
    the outer product of the prediction residual and the penultimate feature, averaged over frames.

    Args:
        proba: Frame probabilities, ``[N, F, C]``.
        last_feature: Penultimate frame features, ``[N, F, H]``.
        xp: Array module to compute in.

    Returns:
        float32 gradient embeddings, ``[N, C, H]``.
    """
    proba = to_backend(proba, xp)
    last_feature = to_backend(last_feature, xp)
    residual = proba - (proba > 0.5).astype(proba.dtype)
    return contract("ntc,ntd->ncd", residual, last_feature) / proba.shape[1]


class BADGE(QueryStrategy):
    """Select a batch from the BADGE gradient embeddings of the unlabeled pool, traversed as the subclass defines.

    Args:
        random_state: Seed or numpy RandomState driving the k-means++ traversal.
    """

    required_properties = ("gradient",)

    def _select(self, ctx):
        gradient = ctx.get("gradient")  # [N, C, H]
        gradient = gradient.reshape(gradient.shape[0], -1)[ctx.unlabeled]  # [n_unlabeled, C*H]
        scores, positions = self._traverse(gradient, ctx.step_size, ctx.xp)
        return scores, ctx.unlabeled[to_numpy(positions)]

    def _traverse(self, gradient, n: int, xp):
        """Return ``(scores, positions)`` of n rows picked from the gradient embeddings ``[n_unlabeled, C*H]``."""
        raise NotImplementedError


class BADGEKMeansPlusPlus(BADGE):
    """Vanilla BADGE, seeding the batch with k-means++ over the gradient embeddings."""

    def _traverse(self, gradient, n, xp):
        return kmeans_plusplus(gradient, n, xp, random_state=self.random_state)


class BADGEMCMCDPP(BADGE):
    """Vanilla BADGE, sampling the batch with the fixed-size swap MCMC DPP of the original repository."""

    def _traverse(self, gradient, n, xp):
        return mcmc_dpp(gradient.astype(xp.float64), n, xp)


class BADGEGreedyDPP(BADGE):
    """The proposed variant, selecting the batch by deterministic greedy maximization of its regularized log-volume."""

    def _traverse(self, gradient, n, xp):
        return greedy_dpp(gradient.astype(xp.float64), n, xp)
