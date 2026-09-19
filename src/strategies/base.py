"""Query interface shared by all strategies, and the per-round context holding prepared model outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends import to_numpy

# Property names a strategy may declare in required_properties, which the learner prepares before each selection.
PROPERTY_NAMES = ("seg_embedding", "prediction_soft", "prediction_hard", "gradient")


# Array fields are compared by object identity.
@dataclass(frozen=True, eq=False)
class QueryContext:
    """Everything a strategy may read while selecting one batch.

    Attributes:
        labeled: Global indices of labeled segments in the order they were selected, int64.
        unlabeled: Global indices of unlabeled segments, int64, sorted ascending.
        step_size: Number of segments to select this round.
        n_train: Size of the training pool.
        frame_embedding: Frame embeddings of the training pool, ``[N, F, D]``, a mmap array read in chunks.
        frame_labels: Frame-level multilabel targets of the training pool, ``[N, F, C]``.
        xp: Array module the prepared properties live in, numpy or cupy.
        properties: Prepared properties keyed by the names in PROPERTY_NAMES.
    """

    labeled: np.ndarray
    unlabeled: np.ndarray
    step_size: int
    n_train: int
    frame_embedding: np.ndarray
    frame_labels: np.ndarray
    xp: object
    properties: dict = field(default_factory=dict)

    def get(self, name: str):
        """Return a prepared property.

        Raises:
            KeyError: The strategy did not declare this property in required_properties.
        """
        try:
            return self.properties[name]
        except KeyError:
            raise KeyError(f"property {name!r} was not prepared; declare it in required_properties") from None

    def labeled_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the labeled frames flattened over segments as ``(embeddings [n, D], targets [n, C])``."""
        embedding = np.asarray(self.frame_embedding[self.labeled])
        labels = np.asarray(self.frame_labels[self.labeled])
        return embedding.reshape(-1, embedding.shape[-1]), labels.reshape(-1, labels.shape[-1])


class QueryStrategy:
    """Base class for segment queries, with subclasses implementing _select.

    Args:
        random_state: Seed or numpy RandomState, which the learner shares across the rounds of an experiment.

    Attributes:
        required_properties: Property names the learner must prepare before the selection step.
    """

    required_properties: tuple[str, ...] = ()

    def __init__(self, *, random_state=None):
        self.random_state = random_state

    def query(self, ctx: QueryContext) -> np.ndarray:
        """Select ``ctx.step_size`` segments and return their global indices, ordered best score first.

        Args:
            ctx: Context carrying the pools and the prepared properties.

        Returns:
            Integer indices, int64 ``[step_size]``.
        """
        scores, idxes = self._select(ctx)
        return self.normalize(scores, idxes)[1]

    def _select(self, ctx: QueryContext):
        """Return ``(scores, global_indices)`` for this round, aligned elementwise."""
        raise NotImplementedError

    def normalize(self, scores, idxes) -> tuple[np.ndarray, np.ndarray]:
        """Move the selection to numpy, fill NaN scores, and order the batch by score, larger being better.

        Args:
            scores: Selection scores in any supported backend.
            idxes: Global indices aligned with scores.

        Returns:
            ``(scores, idxes)`` as float32 and int64 numpy arrays, ordered best first.
        """
        scores = to_numpy(scores).astype(np.float32, copy=False)
        idxes = to_numpy(idxes).astype(np.int64, copy=False)
        return self._sort_by_score(self._replace_nan_scores(scores), idxes)

    def _replace_nan_scores(self, scores: np.ndarray) -> np.ndarray:
        """Replace NaN scores with a value just above the finite maximum, keeping such samples ranked first."""
        nan_mask = np.isnan(scores)
        if not np.any(nan_mask):
            return scores
        finite = scores[~nan_mask]
        if finite.size == 0:
            replacement = 0.0
        else:
            shift = float(finite.astype(np.float64, copy=False).std()) / scores.size
            replacement = float(finite.max()) + shift
        filled = scores.copy()
        filled[nan_mask] = np.float32(replacement)
        return filled

    def _sort_by_score(self, scores: np.ndarray, idxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Order by descending score, breaking ties by the smaller global index so the batch order is deterministic."""
        order = np.lexsort((idxes, -scores))
        return scores[order], idxes[order]
