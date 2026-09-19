"""Test the experiment configuration, property preparation, and the active learning loop on a synthetic cache."""

from __future__ import annotations

import csv
import itertools
import json

import numpy as np
import pytest
import torch

import src.learner as learner_module
import src.strategies.badge as badge_module
from src.data.arrays import load_frame_data
from src.learner import (
    CONFIG_FILENAME,
    CONFIG_HASH_LEN,
    OVERALL_FIELDNAMES,
    RESULTS_FILENAME,
    SELECTION_COLUMN,
    ActiveLearner,
    check_or_write_config,
    config_dir,
    config_hash,
    results_csv_complete,
    strategy_config,
)
from src.models.classifier import FrameMLP, predict_frames
from src.strategies import STRATEGY_NAMES
from src.strategies.badge import badge_gradient
from src.strategies.base import PROPERTY_NAMES

_CONFIG_KWARGS = dict(
    dataset="HyenaSET", strategy="badge-greedy-dpp", step_size=300, max_iter=10, hidden_features=0,
    lr=1e-3, max_epochs=1000, batch_size=32, infer_batch_size=1024, num_classes=10, n_train=51760,
    device="cuda", backend="cuml+cupy",
)


@pytest.fixture
def frame_data(synthetic_cache):
    """Synthetic FrameData with 60 training segments, 8 frames and 3 classes."""
    return load_frame_data(synthetic_cache)


def _learner(frame_data, output_root, **overrides):
    kwargs = dict(
        data=frame_data, dataset="Toy", strategy_name="badge-greedy-dpp", output_root=output_root,
        step_size=4, max_iter=2, hidden_features=8, lr=1e-3, max_epochs=2, seed=0, device="cpu",
        batch_size=4, infer_batch_size=8,
    )
    kwargs.update(overrides)
    return ActiveLearner(**kwargs)


def _read_rows(seed_dir):
    with (seed_dir / RESULTS_FILENAME).open(newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def _rows_without_timing(seed_dir):
    """Drop the wall-clock column, which is the only part of a rerun that legitimately differs."""
    _, rows = _read_rows(seed_dir)
    return [{key: value for key, value in row.items() if key != "query_time"} for row in rows]


def _freeze_clock(monkeypatch) -> None:
    """Make every query stage last exactly one tick, so that reruns can be compared byte for byte."""
    monkeypatch.setattr(learner_module, "perf_counter", itertools.count().__next__)


def _same_weights(state, other) -> bool:
    return all(torch.equal(state[key], other[key]) for key in state)


def _spy_train_round(monkeypatch) -> list[dict]:
    """Record the inputs handed to every training round, the initial state values at that moment, and the result."""
    rounds: list[dict] = []
    original_train_round = learner_module.train_round

    def spy(**kwargs):
        initial_state = kwargs["initial_state"]
        record = dict(
            initial_state=initial_state,
            snapshot={key: value.detach().clone() for key, value in initial_state.items()},
            train_embedding=kwargs["train_embedding"],
            train_label=kwargs["train_label"],
            num_labeled=len(kwargs["train_embedding"]),
        )
        rounds.append(record)
        record["model"], record["round_result"] = original_train_round(**kwargs)
        return record["model"], record["round_result"]

    monkeypatch.setattr(learner_module, "train_round", spy)
    return rounds


def test_config_records_the_implementation_fields():
    config = strategy_config(**_CONFIG_KWARGS)
    assert config["optimizer"] == "adam"
    assert config["model_selection"] == "val_mAP"
    assert config["lr_scheduler"] == "reduce_on_plateau(val_mAP)"
    assert config["early_stopping_patience"] == 10
    assert config["backend"] == "cuml+cupy"  # Recorded as given, never probed from the current process
    assert json.loads(json.dumps(config)) == config


def test_config_hash_is_order_insensitive():
    config = strategy_config(**_CONFIG_KWARGS)
    shuffled = dict(reversed(list(config.items())))
    assert list(shuffled) != list(config)
    assert config_hash(shuffled) == config_hash(config)
    assert len(config_hash(config)) == CONFIG_HASH_LEN == 12


@pytest.mark.parametrize("field,value", [
    ("dataset", "OtherSET"), ("strategy", "badge-mcmc-dpp"), ("step_size", 150), ("max_iter", 9),
    ("hidden_features", 64), ("lr", 2e-3), ("max_epochs", 500), ("batch_size", 64),
    ("infer_batch_size", 512), ("num_classes", 11), ("n_train", 51761), ("device", "cpu"),
    ("backend", "sklearn+numpy"),
])
def test_config_hash_is_sensitive_to_every_field(field, value):
    base = strategy_config(**_CONFIG_KWARGS)
    changed = strategy_config(**{**_CONFIG_KWARGS, field: value})
    assert changed != base and config_hash(changed) != config_hash(base)


def test_config_is_shared_across_seeds(frame_data, tmp_path):
    assert "seed" not in strategy_config(**_CONFIG_KWARGS)
    first, second = _learner(frame_data, tmp_path, seed=0), _learner(frame_data, tmp_path, seed=1)
    assert first.config_dir == second.config_dir
    assert (first.seed_dir.name, second.seed_dir.name) == ("seed_0", "seed_1")


def test_config_dir_layout(tmp_path):
    config = strategy_config(**_CONFIG_KWARGS)
    expected = tmp_path / "HyenaSET" / "badge-greedy-dpp" / config_hash(config)
    assert config_dir(tmp_path, config) == expected and not expected.exists()


def test_check_or_write_config_fails_fast_on_mismatch(tmp_path):
    config = strategy_config(**_CONFIG_KWARGS)
    check_or_write_config(tmp_path / "cfg", config)
    check_or_write_config(tmp_path / "cfg", config)
    assert json.loads((tmp_path / "cfg" / CONFIG_FILENAME).read_text()) == config
    assert [p.name for p in (tmp_path / "cfg").iterdir()] == [CONFIG_FILENAME]  # No leftover tmp file
    with pytest.raises(ValueError, match="does not match"):
        check_or_write_config(tmp_path / "cfg", {**config, "lr": 1.0})


def test_results_csv_complete_checks_the_header_and_the_row_count(tmp_path):
    path = tmp_path / RESULTS_FILENAME
    assert not results_csv_complete(path, 2)

    header = ",".join([*OVERALL_FIELDNAMES, "test_mAP_a", SELECTION_COLUMN])
    path.write_text(f"{header}\n0,4,0.1,,,,,[]\n1,8,0.2,,,,,[]\n")
    assert results_csv_complete(path, 2)
    assert not results_csv_complete(path, 1) and not results_csv_complete(path, 3)

    # A header written by another layout must not be accepted as a resumable run.
    path.write_text("iteration,num_labeled_samples,val_mAP\n0,4,\n1,8,\n")
    assert not results_csv_complete(path, 2)


def test_results_csv_complete_accepts_a_selection_field_past_the_csv_field_limit(tmp_path):
    # The full-supervised reference writes its whole training pool into one field of a single row.
    path = tmp_path / RESULTS_FILENAME
    selection = json.dumps(list(range(60000)))
    assert len(selection) > csv.field_size_limit()
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([*OVERALL_FIELDNAMES, SELECTION_COLUMN])
        writer.writerow([0, 60000, "0.1", "", "", "", selection])
    assert results_csv_complete(path, 1) and not results_csv_complete(path, 2)


def test_prepare_properties_covers_the_whole_training_pool(frame_data, tmp_path):
    learner = _learner(frame_data, tmp_path)
    model = FrameMLP(32, 3, hidden_features=8)
    thresholds = np.array([0.3, 0.5, 0.7], dtype=np.float32)
    properties = learner._prepare_properties(PROPERTY_NAMES, model, thresholds)
    assert set(properties) == set(PROPERTY_NAMES)

    proba, last_feature = predict_frames(
        model, frame_data.train_embedding, batch_size=8, device="cpu", want_features=True
    )
    assert np.array_equal(properties["seg_embedding"], frame_data.seg_embedding)
    assert np.array_equal(properties["prediction_soft"], proba.numpy())
    # The hard prediction applies the per-class thresholds of the selected epoch, not a fixed 0.5.
    hard = properties["prediction_hard"]
    assert hard.dtype == np.int8 and hard.shape == (60, 8, 3)
    assert np.array_equal(hard, (proba.numpy() >= thresholds).astype(np.int8))
    assert not np.array_equal(hard, (proba.numpy() >= 0.5).astype(np.int8))
    # Gradient embeddings are built for every training segment; the strategy slices the unlabeled ones.
    assert properties["gradient"].shape == (60, 3, 8)
    assert np.allclose(properties["gradient"], badge_gradient(proba, last_feature, np), atol=1e-7)


def test_prepare_properties_materializes_only_the_declared_names(frame_data, tmp_path, monkeypatch):
    learner = _learner(frame_data, tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("no inference is needed without a model-based property")

    monkeypatch.setattr(learner_module, "predict_frames", fail_if_called)
    assert learner._prepare_properties((), None, None) == {}
    assert set(learner._prepare_properties(("seg_embedding",), None, None)) == {"seg_embedding"}


def test_learner_rejects_a_budget_larger_than_the_pool(frame_data, tmp_path):
    with pytest.raises(ValueError, match="not enough samples"):
        _learner(frame_data, tmp_path, step_size=31, max_iter=2)


def test_learner_probes_the_backend_only_when_none_is_given(frame_data, tmp_path):
    # The toy learner runs on the CPU, which numpy and scikit-learn serve on any hardware.
    assert _learner(frame_data, tmp_path).config["backend"] == "sklearn+numpy"

    learner = _learner(frame_data, tmp_path, backend="cuml+cupy")
    assert learner.config["backend"] == "cuml+cupy"
    # A process that cannot provide the named backend fails before anything is written.
    with pytest.raises(RuntimeError, match="requires cuml\\+cupy"):
        learner.run()
    assert not learner.config_dir.exists()


def test_learner_writes_the_results_schema_and_no_checkpoints(frame_data, tmp_path):
    learner = _learner(frame_data, tmp_path / "out")
    seed_dir = learner.run()
    assert seed_dir == learner.config_dir / "seed_0"
    assert seed_dir.parent.parent == tmp_path / "out" / "Toy" / "badge-greedy-dpp"
    assert json.loads((seed_dir.parent / CONFIG_FILENAME).read_text()) == learner.config
    # Round weights are not kept, so results.csv is the only file of a seed.
    assert [p.name for p in seed_dir.iterdir()] == [RESULTS_FILENAME]

    fieldnames, rows = _read_rows(seed_dir)
    assert fieldnames == [
        "iteration", "num_labeled_samples", "query_time", "val_mAP", "test_mAP", "test_AULC",
        "test_mAP_class_0", "test_mAP_class_1", "test_mAP_class_2", "queried_idxes_in_latest_iteration",
    ]
    assert fieldnames[:len(OVERALL_FIELDNAMES)] == OVERALL_FIELDNAMES and fieldnames[-1] == SELECTION_COLUMN
    assert [row["iteration"] for row in rows] == ["0", "1"]
    assert [row["num_labeled_samples"] for row in rows] == ["4", "8"]
    assert rows[0]["test_AULC"] == ""  # AULC is undefined for a single point
    # Two equally spaced budgets make the normalized area the mean of the two test mAPs, up to CSV rounding.
    mean_test_map = (float(rows[0]["test_mAP"]) + float(rows[1]["test_mAP"])) / 2
    assert float(rows[1]["test_AULC"]) == pytest.approx(mean_test_map, abs=2e-6)

    labeled: set[int] = set()
    for iteration, row in enumerate(rows):
        selected = json.loads(row[SELECTION_COLUMN])
        assert len(selected) == 4 and len(set(selected)) == 4 and labeled.isdisjoint(selected)
        labeled |= set(selected)
        for key in ("query_time", "val_mAP", "test_mAP", "test_mAP_class_0") + (("test_AULC",) if iteration else ()):
            assert len(row[key].split(".")[1]) == 6
        assert float(row["query_time"]) >= 0.0
        assert 0.0 <= float(row["val_mAP"]) <= 1.0 and 0.0 <= float(row["test_mAP"]) <= 1.0
        per_class = [float(row[f"test_mAP_class_{c}"]) for c in range(3)]
        assert float(row["test_mAP"]) == pytest.approx(np.mean(per_class), abs=2e-6)


def test_learner_resume_skips_complete_and_reruns_incomplete(frame_data, tmp_path):
    seed_dir = _learner(frame_data, tmp_path).run()
    results_csv = seed_dir / RESULTS_FILENAME
    mtime = results_csv.stat().st_mtime_ns
    assert _learner(frame_data, tmp_path).run() == seed_dir
    assert results_csv.stat().st_mtime_ns == mtime

    complete = _rows_without_timing(seed_dir)
    text = results_csv.read_text()
    results_csv.write_text("".join(text.splitlines(keepends=True)[:2]))
    (seed_dir / "stale.txt").write_text("x")
    _learner(frame_data, tmp_path).run()
    assert _rows_without_timing(seed_dir) == complete
    assert not (seed_dir / "stale.txt").exists()


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_learner_rerun_with_the_same_seed_is_byte_identical(frame_data, tmp_path, monkeypatch, strategy):
    # max_iter=3 gives every strategy two model-driven rounds, so a stream replayed or left unseeded would show.
    _freeze_clock(monkeypatch)
    first = _learner(frame_data, tmp_path / "a", strategy_name=strategy, max_iter=3).run()
    second = _learner(frame_data, tmp_path / "b", strategy_name=strategy, max_iter=3).run()
    assert (first / RESULTS_FILENAME).read_bytes() == (second / RESULTS_FILENAME).read_bytes()


def test_learner_separates_seeds(frame_data, tmp_path):
    seed_0 = _learner(frame_data, tmp_path, seed=0).run()
    seed_1 = _learner(frame_data, tmp_path, seed=1).run()
    assert seed_1.parent == seed_0.parent and (seed_0.name, seed_1.name) == ("seed_0", "seed_1")
    # The seed drives the cold start and the initial weights, so the two runs differ from round 0 on.
    assert _rows_without_timing(seed_0)[0] != _rows_without_timing(seed_1)[0]


def test_learner_cold_start_is_shared_across_strategies(frame_data, tmp_path, monkeypatch):
    _freeze_clock(monkeypatch)
    rows = {
        strategy: _read_rows(_learner(frame_data, tmp_path, strategy_name=strategy).run())[1]
        for strategy in ("badge-greedy-dpp", "badge-kmeans++", "entropy")
    }
    # Round 0 draws the shared cold-start set and trains from the same initial weights, so its whole row
    # matches under the same seed; the strategies part ways only from round 1 on.
    assert rows["badge-greedy-dpp"][0] == rows["badge-kmeans++"][0] == rows["entropy"][0]
    second_rounds = {tuple(json.loads(strategy_rows[1][SELECTION_COLUMN])) for strategy_rows in rows.values()}
    assert len(second_rounds) == 3


def test_learner_retrains_every_round_from_the_same_initial_weights(frame_data, tmp_path, monkeypatch):
    rounds = _spy_train_round(monkeypatch)
    _learner(frame_data, tmp_path, max_iter=3).run()
    assert [record["num_labeled"] for record in rounds] == [4, 8, 12]
    for record in rounds:
        # One state object serves the whole experiment, and no round trains it in place.
        assert record["initial_state"] is rounds[0]["initial_state"]
        assert _same_weights(record["snapshot"], rounds[0]["snapshot"])
        assert not _same_weights(record["model"].state_dict(), rounds[0]["snapshot"])
    assert len({id(record["model"]) for record in rounds}) == 3  # A fresh head is built every round.


def test_learner_trains_on_device_tensors_in_the_order_of_selection(frame_data, tmp_path, monkeypatch):
    rounds = _spy_train_round(monkeypatch)
    seed_dir = _learner(frame_data, tmp_path, strategy_name="entropy", max_iter=3).run()
    selections = [json.loads(row[SELECTION_COLUMN]) for row in _read_rows(seed_dir)[1]]
    # Entropy orders a batch by score, so the labeled set is not sorted and its order is observable.
    assert any(batch != sorted(batch) for batch in selections)
    for iteration, record in enumerate(rounds):
        labeled = np.concatenate(selections[:iteration + 1])
        assert torch.is_tensor(record["train_embedding"]) and torch.is_tensor(record["train_label"])
        assert torch.equal(record["train_embedding"], torch.from_numpy(frame_data.train_embedding[labeled]))
        assert torch.equal(record["train_label"], torch.from_numpy(frame_data.train_label[labeled]))


def test_learner_initial_weights_follow_the_seed_not_the_strategy(frame_data, tmp_path, monkeypatch):
    rounds = _spy_train_round(monkeypatch)
    _learner(frame_data, tmp_path, strategy_name="entropy", max_iter=1).run()
    _learner(frame_data, tmp_path, strategy_name="random", max_iter=1).run()
    _learner(frame_data, tmp_path, strategy_name="entropy", max_iter=1, seed=1).run()
    entropy_seed_0, random_seed_0, entropy_seed_1 = (record["snapshot"] for record in rounds)
    assert _same_weights(entropy_seed_0, random_seed_0)
    assert not _same_weights(entropy_seed_0, entropy_seed_1)


def test_learner_queries_with_the_model_selected_in_the_previous_round(frame_data, tmp_path, monkeypatch):
    rounds = _spy_train_round(monkeypatch)
    queried = []
    original_prepare = ActiveLearner._prepare_properties

    def spy_prepare(self, names, model, thresholds):
        queried.append((model, thresholds))
        return original_prepare(self, names, model, thresholds)

    monkeypatch.setattr(ActiveLearner, "_prepare_properties", spy_prepare)
    _learner(frame_data, tmp_path, strategy_name="disagreement", max_iter=3).run()
    assert queried[0] == (None, None)  # The cold start has no model yet.
    for (model, thresholds), previous in zip(queried[1:], rounds):
        assert model is previous["model"] and thresholds is previous["round_result"].thresholds


def test_learner_kmeans_rounds_advance_one_shared_random_stream(frame_data, tmp_path, monkeypatch):
    def stream_position(random_state):
        _, keys, position, *_ = random_state.get_state()
        return keys.tobytes(), position

    seen = []
    original_kmeans_plusplus = badge_module.kmeans_plusplus

    def spy(X, n, xp, *, random_state):
        seen.append((random_state, stream_position(random_state)))
        return original_kmeans_plusplus(X, n, xp, random_state=random_state)

    monkeypatch.setattr(badge_module, "kmeans_plusplus", spy)
    learner = _learner(frame_data, tmp_path, strategy_name="badge-kmeans++", max_iter=3)
    learner.run()
    # Round 0 is the cold start, so k-means++ runs in rounds 1 and 2. The strategy is rebuilt every round
    # but keeps drawing from the stream of the experiment, so round 2 does not replay the draws of round 1.
    (first_stream, first_position), (second_stream, second_position) = seen
    assert isinstance(first_stream, np.random.RandomState)
    assert first_stream is second_stream is learner.random_state
    assert first_position != second_position


def test_query_time_covers_property_preparation_and_selection_only(frame_data, tmp_path, monkeypatch):
    # A manual clock advanced by the spied stages makes the measured span exact.
    clock = {"now": 0.0}
    monkeypatch.setattr(learner_module, "perf_counter", lambda: clock["now"])

    prepared = []
    original_prepare = ActiveLearner._prepare_properties

    def slow_prepare(self, names, model, thresholds):
        prepared.append(tuple(names))
        clock["now"] += 2.0
        return original_prepare(self, names, model, thresholds)

    original_build = learner_module.build_query_strategy

    def build_slow_strategy(name, *, random_state):
        strategy = original_build(name, random_state=random_state)
        original_query = strategy.query

        def slow_query(ctx):
            clock["now"] += 3.0
            return original_query(ctx)

        strategy.query = slow_query
        return strategy

    original_train_round = learner_module.train_round

    def slow_train_round(**kwargs):
        clock["now"] += 100.0
        return original_train_round(**kwargs)

    monkeypatch.setattr(ActiveLearner, "_prepare_properties", slow_prepare)
    monkeypatch.setattr(learner_module, "build_query_strategy", build_slow_strategy)
    monkeypatch.setattr(learner_module, "train_round", slow_train_round)

    seed_dir = _learner(frame_data, tmp_path, strategy_name="mfft").run()
    # Round 0 falls back to the cold start, which declares nothing, so only round 1 prepares properties.
    assert prepared == [(), ("seg_embedding", "prediction_hard")]
    _, rows = _read_rows(seed_dir)
    assert [row["query_time"] for row in rows] == ["5.000000", "5.000000"]


def test_learner_rejects_an_invalid_selection(frame_data, tmp_path, monkeypatch):
    class DuplicateStrategy:
        required_properties = ()

        def query(self, ctx):
            return np.full(ctx.step_size, ctx.unlabeled[0], dtype=np.int64)

    monkeypatch.setattr(
        learner_module, "build_query_strategy", lambda name, *, random_state: DuplicateStrategy()
    )
    with pytest.raises(RuntimeError, match="invalid selection"):
        _learner(frame_data, tmp_path).run()


@pytest.mark.parametrize("value,expected", [
    (0.5, "0.500000"), (1.0 / 3.0, "0.333333"), (np.float64(2.0), "2.000000"), (float("nan"), ""),
])
def test_metric_formatting_uses_six_decimals_and_blanks_undefined_values(value, expected):
    assert ActiveLearner._fmt(value) == expected
