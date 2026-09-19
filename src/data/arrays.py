"""Share precomputed arrays through copy-on-write mmap, keeping the disk cache unchanged."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_FILENAME = "meta.json"

# npy file stems of the FrameData fields. Other files of the cache are not read.
ARRAY_FILES = {
    "train_embedding": "train_embedding",
    "train_label": "train_label",
    "seg_embedding": "train_seg_embedding",
    "val_embedding": "val_embedding",
    "val_label": "val_label",
    "test_embedding": "test_embedding",
    "test_label": "test_label",
}


# Array fields are compared by object identity.
@dataclass(frozen=True, eq=False)
class FrameData:
    """Arrays and class names for a single dataset.

    Attributes:
        train_embedding: Training frame embeddings, float32 ``[N, F, D]``.
        train_label: Training frame multilabel targets, float32 ``[N, F, C]``.
        seg_embedding: Training segment representations, the mean over the frames, float32 ``[N, D]``.
        val_embedding: Validation frame embeddings, ``[M, F, D]``.
        val_label: Validation frame multilabel targets, ``[M, F, C]``.
        test_embedding: Test frame embeddings, ``[K, F, D]``.
        test_label: Test frame multilabel targets, ``[K, F, C]``.
        class_names: Class names, ordered to match the last label dimension.
    """

    train_embedding: np.ndarray
    train_label: np.ndarray
    seg_embedding: np.ndarray
    val_embedding: np.ndarray
    val_label: np.ndarray
    test_embedding: np.ndarray
    test_label: np.ndarray
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        """Number of classes, the length of class_names."""
        return len(self.class_names)

    @property
    def n_train(self) -> int:
        """Number of segments in the training pool."""
        return self.train_embedding.shape[0]


def _read_meta(cache_dir: Path) -> dict:
    """Parse meta.json of a dataset cache directory."""
    return json.loads((cache_dir / META_FILENAME).read_text(encoding="utf-8"))


def read_array_meta(cache_dir) -> tuple[int, int]:
    """Read ``(n_train, num_classes)`` from meta.json without loading arrays."""
    meta = _read_meta(Path(cache_dir))
    return meta["shapes"]["train_embedding"][0], len(meta["class_names"])


def load_frame_data(cache_dir) -> FrameData:
    """Load arrays and class names for a single dataset with ``mmap_mode="c"``."""
    cache_dir = Path(cache_dir)
    meta = _read_meta(cache_dir)
    arrays = {
        field: np.load(cache_dir / f"{stem}.npy", mmap_mode="c")
        for field, stem in ARRAY_FILES.items()
    }
    return FrameData(**arrays, class_names=tuple(meta["class_names"]))
