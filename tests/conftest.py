"""Build a synthetic frame-level cache in the real on-disk format and share it as a fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Limit threads before importing numpy/torch to prevent CPU fallback from crashing on many-core nodes.
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rootutils  # noqa: E402

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

# Frame prevalence of the first class and the geometric decay applied to the following ones, so the last
# classes stand in for the rare call types of the paper.
HEAD_CLASS_RATE = 0.45
TAIL_DECAY = 0.3


def _mean_pool(frame_embedding: np.ndarray) -> np.ndarray:
    """Average the frames of each segment in float64, then convert back to float32."""
    return frame_embedding.astype(np.float64).mean(axis=1).astype(np.float32)


def build_synthetic_frame_cache(
    cache_dir,
    *,
    n_train: int = 60,
    n_val: int = 16,
    n_test: int = 16,
    num_classes: int = 3,
    num_frames: int = 8,
    embedding_dim: int = 32,
    num_clusters: int = 4,
    seed: int = 0,
) -> Path:
    """Write a clustered, long-tailed synthetic cache and return cache_dir.

    Segments are drawn around ``num_clusters`` centers and a class only occurs inside some of them, so a
    segment label is predictable from its embedding and the neighbour-based strategies are meaningful.
    Frame prevalence decays geometrically over the classes, which keeps the last classes rare.

    Args:
        cache_dir: Directory the npy arrays and meta.json are written to.
        n_train: Number of training segments.
        n_val: Number of validation segments.
        n_test: Number of test segments.
        num_classes: Number of classes.
        num_frames: Frames per segment.
        embedding_dim: Frame embedding dimension.
        num_clusters: Number of embedding clusters shared by the three splits.
        seed: Seed of the generator producing the arrays.

    Returns:
        The cache directory, as a Path.
    """
    rng = np.random.default_rng(seed)
    class_rate = HEAD_CLASS_RATE * TAIL_DECAY ** np.arange(num_classes)
    centers = rng.standard_normal((num_clusters, embedding_dim)).astype(np.float32) * 3.0
    cluster_active = rng.random((num_clusters, num_classes)) < 0.5

    def make_split(num_segments: int) -> tuple[np.ndarray, np.ndarray]:
        cluster = rng.integers(num_clusters, size=num_segments)
        embedding = (
            centers[cluster][:, None, :]
            + rng.standard_normal((num_segments, num_frames, embedding_dim)).astype(np.float32)
        )
        active = rng.random((num_segments, num_frames, num_classes)) < class_rate
        label = (active & cluster_active[cluster][:, None, :]).astype(np.float32)
        for c in range(num_classes):  # Each class keeps at least one positive frame so that mAP is defined.
            if label[..., c].sum() == 0:
                label[c % num_segments, 0, c] = 1.0
        return embedding.astype(np.float32), label

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, list[int]] = {}

    def save(name: str, array: np.ndarray) -> None:
        np.save(cache_dir / f"{name}.npy", array)
        shapes[name] = list(array.shape)

    for split, num_segments in (("train", n_train), ("val", n_val), ("test", n_test)):
        embedding, label = make_split(num_segments)
        save(f"{split}_embedding", embedding)
        save(f"{split}_label", label)
        if split == "train":
            save("train_seg_embedding", _mean_pool(embedding))

    (cache_dir / "meta.json").write_text(
        json.dumps({
            "class_names": [f"class_{c}" for c in range(num_classes)],
            "shapes": shapes,
        }, indent=2),
        encoding="utf-8",
    )
    return cache_dir


@pytest.fixture
def synthetic_cache(tmp_path) -> Path:
    """Build the default synthetic cache at tmp_path/cache/Toy."""
    return build_synthetic_frame_cache(tmp_path / "cache" / "Toy")
