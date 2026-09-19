"""Nearest-neighbour auxiliary classifier labelling unlabeled frames from the labeled frames."""

from __future__ import annotations

import numpy as np

from .backends import is_cupy, to_backend, to_numpy

# Segments per neighbour query chunk, which bounds device memory.
QUERY_CHUNK_SEGMENTS = 2048


def nearest_labeled_targets(labeled_features, labeled_targets, query_frames, query_rows, xp) -> np.ndarray:
    """Copy the target of the nearest labeled frame onto every frame of the query segments.

    Args:
        labeled_features: Labeled frame embeddings, ``[n_lab, D]``.
        labeled_targets: Labeled frame multilabel targets, ``[n_lab, C]``.
        query_frames: Frame embeddings of the whole pool, ``[N, F, D]``, typically a mmap array.
        query_rows: Rows of query_frames to score, ``[n_query]``.
        xp: Array module the neighbour search runs in, served by cuml for cupy and by scikit-learn otherwise.

    Returns:
        int8 reference targets, ``[n_query, F, C]``.
    """
    if is_cupy(xp):
        from cuml.neighbors import NearestNeighbors
    else:
        from sklearn.neighbors import NearestNeighbors

        # scikit-learn only takes host arrays.
        xp = np

    estimator = NearestNeighbors(n_neighbors=1)
    estimator.fit(to_backend(labeled_features, xp))
    targets = (np.asarray(labeled_targets) >= 0.5).astype(np.int8)

    num_frames, num_features = query_frames.shape[1], query_frames.shape[2]
    num_segments = len(query_rows)
    reference = np.empty((num_segments, num_frames, targets.shape[1]), dtype=np.int8)
    for start in range(0, num_segments, QUERY_CHUNK_SEGMENTS):
        stop = min(start + QUERY_CHUNK_SEGMENTS, num_segments)
        chunk = to_backend(
            np.asarray(query_frames[query_rows[start:stop]]).reshape(-1, num_features), xp
        )
        indices = estimator.kneighbors(chunk, return_distance=False)
        reference[start:stop] = targets[to_numpy(indices).astype(np.int64).reshape(-1)].reshape(
            stop - start, num_frames, -1
        )
    return reference
