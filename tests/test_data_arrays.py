"""Test cache metadata, FrameData properties, and copy-on-write mmap."""

from __future__ import annotations

import numpy as np

from src.data.arrays import ARRAY_FILES, FrameData, load_frame_data, read_array_meta


def test_load_frame_data_matches_files_on_disk(synthetic_cache):
    data = load_frame_data(synthetic_cache)
    assert isinstance(data, FrameData)
    for field, stem in ARRAY_FILES.items():
        expected = np.load(synthetic_cache / f"{stem}.npy")
        actual = getattr(data, field)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert np.asarray(actual).tobytes() == expected.tobytes()


def test_load_frame_data_ignores_other_files_of_the_cache(synthetic_cache):
    # The published cache also ships arrays the code does not read, such as the valid frame counts.
    np.save(synthetic_cache / "train_valid_lengths.npy", np.full(60, 8, dtype=np.int64))
    data = load_frame_data(synthetic_cache)
    assert not hasattr(data, "train_valid_lengths")
    assert set(ARRAY_FILES) == {
        "train_embedding", "train_label", "seg_embedding", "val_embedding", "val_label",
        "test_embedding", "test_label",
    }


def test_frame_data_properties_match_array_shapes(synthetic_cache):
    data = load_frame_data(synthetic_cache)
    assert (data.n_train, data.num_classes) == (60, 3)
    assert data.class_names == ("class_0", "class_1", "class_2")
    assert data.train_embedding.shape == (data.n_train, 8, 32)
    assert data.seg_embedding.shape == (data.n_train, 32)
    assert data.val_embedding.shape[1:] == data.test_embedding.shape[1:] == (8, 32)


def test_read_array_meta_matches_loaded_data(synthetic_cache):
    data = load_frame_data(synthetic_cache)
    assert read_array_meta(synthetic_cache) == (data.n_train, data.num_classes)


def test_load_frame_data_is_copy_on_write_mmap(synthetic_cache):
    data = load_frame_data(synthetic_cache)
    assert isinstance(data.train_embedding, np.memmap)
    assert data.train_embedding.flags.writeable
    original = float(data.train_embedding[0, 0, 0])
    data.train_embedding[0, 0, 0] = original + 1.0
    assert float(np.load(synthetic_cache / "train_embedding.npy")[0, 0, 0]) == original


def test_frame_labels_are_long_tailed(synthetic_cache):
    data = load_frame_data(synthetic_cache)
    prevalence = np.asarray(data.train_label).mean(axis=(0, 1))
    assert (prevalence > 0.0).all() and prevalence[-1] < prevalence[0]
