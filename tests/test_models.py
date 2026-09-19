"""Test the classification head, its training round, and the evaluation metrics."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_precision_recall_curve

import src.models.classifier as classifier_module
from src.data.arrays import load_frame_data
from src.models.classifier import (
    FrameMLP,
    RoundResult,
    default_hidden_features,
    initial_head_state,
    predict_frames,
    train_round,
)
from src.models.metrics import (
    average_precision_tensors,
    cumulative_aulc,
    frame_wise_map,
    mean_average_precision,
    precision_recall_curves,
)


@pytest.mark.parametrize("input_dim,num_classes,expected", [
    (768, 10, 64),   # Paper protocol: animal2vec frames and ten call types
    (2048, 10, 128),
    (32, 3, 64),     # Below the lower bound, so the clamp applies
    (8192, 64, 704),
    (65536, 4096, 2048),  # Above the upper bound, so the clamp applies
])
def test_default_hidden_features_rounds_the_clamped_geometric_mean(input_dim, num_classes, expected):
    assert default_hidden_features(input_dim, num_classes) == expected


def test_frame_mlp_output_shapes_and_default_width():
    model = FrameMLP(768, 10)
    logit, last_feature = model(torch.zeros(4, 20, 768))
    assert model.hidden_features == 64
    assert logit.shape == (4, 20, 10) and last_feature.shape == (4, 20, 64)


def test_frame_mlp_last_feature_is_the_post_norm_pre_fc2_activation():
    model = FrameMLP(16, 3, hidden_features=8).eval()  # Dropout is identity in eval, so the path is exact.
    x = torch.randn(5, 6, 16)
    logit, last_feature = model(x)
    hidden = model.act(model.fc1(x))
    expected = model.norm(hidden.transpose(-1, -2)).transpose(-1, -2)
    assert torch.allclose(last_feature, expected, atol=1e-6)
    assert torch.allclose(logit, model.fc2(last_feature), atol=1e-6)


def test_frame_mlp_drops_a_quarter_of_the_logits_while_training():
    torch.manual_seed(0)
    model = FrameMLP(16, 3, hidden_features=8).train()
    logit, _ = model(torch.randn(64, 6, 16))
    # Dropout after the output layer zeroes whole logits, which a plain linear layer never produces.
    assert 0.18 < float((logit == 0.0).float().mean()) < 0.32


def test_initial_head_state_matches_the_head_layout():
    state = initial_head_state(16, 3, 8, "cpu")
    assert set(state) == set(FrameMLP(16, 3, 8).state_dict())
    assert state["fc1.weight"].shape == (8, 16) and state["fc2.weight"].shape == (3, 8)
    # The default width rule applies when no hidden width is given.
    assert initial_head_state(768, 10, None, "cpu")["fc2.weight"].shape == (10, 64)


def test_initial_head_state_follows_the_torch_seed():
    torch.manual_seed(0)
    first = initial_head_state(16, 3, 8, "cpu")
    torch.manual_seed(0)
    again = initial_head_state(16, 3, 8, "cpu")
    torch.manual_seed(1)
    other = initial_head_state(16, 3, 8, "cpu")
    assert all(torch.equal(first[key], again[key]) for key in first)
    assert not torch.equal(first["fc1.weight"], other["fc1.weight"])


def test_predict_frames_batches_match_a_single_pass():
    model = FrameMLP(16, 3, hidden_features=8).train()
    embeddings = np.random.default_rng(0).standard_normal((7, 6, 16)).astype(np.float32)
    proba = predict_frames(model, embeddings, batch_size=3, device="cpu")
    assert not model.training  # Inference must not run with dropout or batch statistics.
    assert proba.shape == (7, 6, 3) and proba.dtype == torch.float32
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0
    assert torch.allclose(proba, predict_frames(model, embeddings, batch_size=7, device="cpu"), atol=1e-6)

    proba_again, last_feature = predict_frames(
        model, embeddings, batch_size=3, device="cpu", want_features=True
    )
    assert last_feature.shape == (7, 6, 8) and torch.equal(proba, proba_again)


def test_predict_frames_accepts_a_mmap_array_and_a_device_tensor(synthetic_cache):
    data = load_frame_data(synthetic_cache)
    model = FrameMLP(32, 3, hidden_features=8)
    from_mmap = predict_frames(model, data.val_embedding, batch_size=5, device="cpu")
    on_device = torch.from_numpy(np.ascontiguousarray(data.val_embedding))
    assert from_mmap.shape == (16, 8, 3)
    assert torch.equal(from_mmap, predict_frames(model, on_device, batch_size=5, device="cpu"))


def _tied_frame_scores(num_frames=400, num_classes=5, seed=0):
    """Return probabilities quantized to a few levels, so every class is full of tied scores."""
    rng = np.random.default_rng(seed)
    proba = torch.from_numpy((rng.integers(0, 12, size=(num_frames, num_classes)) / 11.0).astype(np.float32))
    targets = torch.from_numpy((rng.random((num_frames, num_classes)) < [0.5, 0.3, 0.1, 0.02, 0.9]).astype(np.int64))
    return proba, targets


@pytest.mark.parametrize("make_inputs", [
    lambda: _tied_frame_scores(),
    lambda: (torch.rand(300, 4, generator=torch.Generator().manual_seed(0)),
             (torch.rand(300, 4, generator=torch.Generator().manual_seed(1)) < 0.2).long()),
], ids=["tied", "continuous"])
def test_average_precision_matches_torchmetrics(make_inputs):
    proba, targets = make_inputs()
    expected = multilabel_average_precision(proba, targets, num_labels=targets.shape[1], average=None)
    per_class, macro = mean_average_precision(precision_recall_curves(proba, targets))
    assert per_class.dtype == np.float64
    assert np.array_equal(per_class, expected.numpy().astype(np.float64))
    # The macro average is the float32 mean over the classes, as torchmetrics reduces it.
    assert macro == pytest.approx(float(expected.mean()), abs=1e-7)
    assert macro == float(average_precision_tensors(precision_recall_curves(proba, targets))[1])


def test_frame_wise_map_composes_the_curves_and_their_reduction():
    proba, targets = _tied_frame_scores(seed=1)
    per_class, macro = frame_wise_map(proba, targets)
    expected_per_class, expected_macro = mean_average_precision(precision_recall_curves(proba, targets))
    assert np.array_equal(per_class, expected_per_class) and macro == expected_macro


def test_frame_wise_map_matches_a_hand_computed_case():
    # Class 0 ranks the four frames 0.9 > 0.8 > 0.7 > 0.6 with targets 1, 0, 1, 0, so the average
    # precision is (1/1 + 2/3) / 2 = 5/6. Class 1 has no positive frame at all.
    proba = torch.tensor([[0.9, 0.5], [0.8, 0.5], [0.7, 0.5], [0.6, 0.5]])
    targets = torch.tensor([[1, 0], [0, 0], [1, 0], [0, 0]])
    per_class, macro = frame_wise_map(proba, targets)
    assert per_class.dtype == np.float64 and per_class.shape == (2,)
    assert per_class[0] == pytest.approx(5.0 / 6.0, abs=1e-6)
    assert per_class[1] == 0.0  # A class without positive frames scores zero, not NaN
    assert macro == pytest.approx(5.0 / 12.0, abs=1e-6)


def test_frame_wise_map_all_classes_absent_is_zero():
    per_class, macro = frame_wise_map(torch.full((5, 2), 0.5), torch.zeros((5, 2), dtype=torch.int64))
    assert (per_class == 0.0).all() and macro == 0.0


@pytest.mark.parametrize("xs,ys", [([300], [0.5]), ([], [])])
def test_cumulative_aulc_is_nan_with_fewer_than_two_points(xs, ys):
    assert math.isnan(cumulative_aulc(xs, ys))


def test_cumulative_aulc_normalizes_by_the_budget_span():
    assert cumulative_aulc([300, 600], [0.4, 0.6]) == pytest.approx(0.5)
    assert cumulative_aulc([300, 600, 900], [0.2, 0.4, 0.6]) == pytest.approx(0.4)
    # A constant curve integrates to its own level whatever the spacing of the budgets.
    assert cumulative_aulc([300, 600, 3000], [0.7, 0.7, 0.7]) == pytest.approx(0.7)


def _toy_split(num_segments, num_classes, *, num_frames=6, dim=16, seed=0):
    """Return random frame embeddings and sparse frame targets of one split."""
    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal((num_segments, num_frames, dim)).astype(np.float32)
    label = (rng.random((num_segments, num_frames, num_classes)) > 0.7).astype(np.float32)
    label[0, 0] = 1.0  # Every class keeps a positive frame so that the validation mAP is defined.
    return embedding, label


def _train_kwargs(num_classes=3, **overrides):
    x_train, y_train = _toy_split(8, num_classes)
    x_val, y_val = _toy_split(6, num_classes, seed=1)
    kwargs = dict(
        train_embedding=torch.from_numpy(x_train), train_label=torch.from_numpy(y_train),
        val_embedding=torch.from_numpy(x_val),
        val_targets=torch.from_numpy(y_val.reshape(-1, num_classes).astype(np.int64)),
        initial_state=initial_head_state(16, num_classes, 8, "cpu"), num_classes=num_classes,
        hidden_features=8, lr=1e-3, max_epochs=6, batch_size=4, infer_batch_size=4, device="cpu",
    )
    kwargs.update(overrides)
    return kwargs


def _same_weights(state, other) -> bool:
    return all(torch.equal(state[key], other[key]) for key in state)


def test_round_result_only_carries_the_selected_epoch_summary():
    assert [f.name for f in dataclasses.fields(RoundResult)] == ["thresholds", "val_map", "num_epochs"]


def test_train_round_returns_the_weights_of_the_reported_validation_map():
    kwargs = _train_kwargs()
    model, round_result = train_round(**kwargs)
    assert isinstance(model, FrameMLP) and round_result.num_epochs == 6
    assert round_result.thresholds.shape == (3,) and round_result.thresholds.dtype == np.float32
    assert ((round_result.thresholds >= 0.0) & (round_result.thresholds <= 1.0)).all()
    # The returned head is already restored to the selected epoch, so it reproduces the reported val mAP
    # over all validation frames.
    val_proba = predict_frames(model, kwargs["val_embedding"], batch_size=4, device="cpu").reshape(-1, 3)
    assert frame_wise_map(val_proba, kwargs["val_targets"])[1] == round_result.val_map


def test_train_round_starts_from_the_initial_state_and_leaves_it_untouched():
    kwargs = _train_kwargs(lr=0.0, max_epochs=2)
    initial_state = kwargs["initial_state"]
    snapshot = {key: value.clone() for key, value in initial_state.items()}
    model, _ = train_round(**kwargs)
    # A zero learning rate freezes the parameters, so they must still be the ones that were handed in.
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, snapshot[name])

    trained, _ = train_round(**{**kwargs, "lr": 1e-2})
    assert not torch.equal(trained.fc1.weight, snapshot["fc1.weight"])
    assert _same_weights(initial_state, snapshot)  # Training works on a copy, never on the shared state.


def test_train_round_repeats_from_the_same_initial_state_and_torch_seed():
    kwargs = _train_kwargs()
    torch.manual_seed(3)
    first, first_round = train_round(**kwargs)
    torch.manual_seed(3)
    second, second_round = train_round(**kwargs)
    assert _same_weights(first.state_dict(), second.state_dict())
    assert first_round.val_map == second_round.val_map
    assert np.array_equal(first_round.thresholds, second_round.thresholds)


def _drive_validation_maps(monkeypatch, val_maps, *, epoch_thresholds=None, epoch_f1s=None) -> list[dict]:
    """Replace the validation summary by fixed sequences and record the weights validated at every epoch."""
    epoch_thresholds = epoch_thresholds or [np.full(3, 0.5, dtype=np.float32)] * len(val_maps)
    epoch_f1s = epoch_f1s or [0.0] * len(val_maps)
    remaining = iter(zip(val_maps, epoch_thresholds, epoch_f1s))
    snapshots: list[dict] = []
    original_predict = classifier_module.predict_frames

    def recording_predict(model, *args, **kwargs):
        snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return original_predict(model, *args, **kwargs)

    monkeypatch.setattr(classifier_module, "predict_frames", recording_predict)
    monkeypatch.setattr(classifier_module, "_validation_summary", lambda proba, targets: next(remaining))
    return snapshots


@pytest.mark.parametrize("val_maps,patience,expected_epochs,selected_epoch", [
    # A later epoch that only ties the best one is not an improvement, so the earliest best epoch is kept.
    ([0.3, 0.5, 0.5, 0.4, 0.2, 0.9], 3, 5, 1),
    # An improvement resets the patience counter.
    ([0.2, 0.1, 0.3, 0.1, 0.1, 0.9], 2, 5, 2),
    # Training runs to max_epochs while the validation mAP keeps improving.
    ([0.1, 0.2, 0.3, 0.4, 0.5], 2, 5, 4),
])
def test_train_round_selects_and_early_stops_on_validation_map(
    monkeypatch, val_maps, patience, expected_epochs, selected_epoch
):
    snapshots = _drive_validation_maps(monkeypatch, val_maps)
    monkeypatch.setattr(classifier_module, "EARLY_STOPPING_PATIENCE", patience)
    model, round_result = train_round(**_train_kwargs(max_epochs=len(val_maps)))

    assert round_result.num_epochs == expected_epochs == len(snapshots)
    assert round_result.val_map == max(val_maps[:expected_epochs])
    assert _same_weights(model.state_dict(), snapshots[selected_epoch])
    for epoch, snapshot in enumerate(snapshots):
        if epoch != selected_epoch:
            assert not _same_weights(model.state_dict(), snapshot)


def test_train_round_default_patience_is_ten_epochs(monkeypatch):
    _drive_validation_maps(monkeypatch, [0.5] + [0.1] * 29)
    _, round_result = train_round(**_train_kwargs(max_epochs=30))
    assert classifier_module.EARLY_STOPPING_PATIENCE == 10
    assert round_result.num_epochs == 11 and round_result.val_map == 0.5


def test_train_round_anneals_the_learning_rate_on_validation_map(monkeypatch):
    created, stepped = [], []

    class RecordingScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def __init__(self, optimizer, **kwargs):
            created.append(kwargs)
            super().__init__(optimizer, **kwargs)

        def step(self, metrics, *args, **kwargs):
            stepped.append(metrics)
            super().step(metrics, *args, **kwargs)

    val_maps = [0.3, 0.5, 0.4]
    _drive_validation_maps(monkeypatch, val_maps)
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", RecordingScheduler)
    train_round(**_train_kwargs(max_epochs=3))
    assert created == [dict(mode="max", factor=0.1, patience=5, min_lr=1e-6)]
    assert stepped == val_maps


def test_best_f1_thresholds_match_a_hand_computed_case():
    # Class 0 separates perfectly at 0.8 and class 1 at 0.6, the lowest score that is still a positive.
    proba = torch.tensor([[0.9, 0.1], [0.8, 0.7], [0.3, 0.6], [0.2, 0.4]])
    targets = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]])
    thresholds, mean_f1 = classifier_module._best_f1_thresholds(precision_recall_curves(proba, targets))
    assert thresholds.dtype == torch.float32 and torch.allclose(thresholds, torch.tensor([0.8, 0.6]))
    assert float(mean_f1) == pytest.approx(1.0, abs=1e-6)


def _separate_validation_passes(proba, targets):
    """Compute the epoch summary the costly way, with one curve pass for the mAP and another for the thresholds."""
    val_map = float(multilabel_average_precision(proba, targets, num_labels=targets.shape[1], average=None).mean())
    best_thresholds, best_f1_scores = [], []
    for p, r, t in zip(*multilabel_precision_recall_curve(proba, targets, num_labels=targets.shape[1])):
        f1 = torch.nan_to_num((2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-8), nan=0.0)
        best_thresholds.append(t[torch.argmax(f1)])
        best_f1_scores.append(f1.max())
    return val_map, torch.stack(best_thresholds).numpy(), float(torch.stack(best_f1_scores).mean())


@pytest.mark.parametrize("seed", range(5))
def test_validation_summary_is_bit_identical_to_separate_curve_passes(seed):
    proba, targets = _tied_frame_scores(seed=seed)
    val_map, thresholds, mean_f1 = classifier_module._validation_summary(proba, targets)
    expected_map, expected_thresholds, expected_f1 = _separate_validation_passes(proba, targets)
    assert val_map == expected_map and mean_f1 == expected_f1
    assert thresholds.dtype == np.float32 and np.array_equal(thresholds, expected_thresholds)


def test_train_round_computes_the_validation_curves_once_per_epoch(monkeypatch):
    calls = []
    original = classifier_module.precision_recall_curves

    def counting(proba, targets):
        calls.append(proba.shape)
        return original(proba, targets)

    monkeypatch.setattr(classifier_module, "precision_recall_curves", counting)
    _, round_result = train_round(**_train_kwargs(max_epochs=3))
    # Every validation frame is scored, and one pass serves both the mAP and the thresholds.
    assert calls == [(6 * 6, 3)] * round_result.num_epochs


@pytest.mark.parametrize("val_maps,epoch_f1s,threshold_epoch", [
    # Epoch 1 is selected, yet its own F1 is lower than that of epoch 0, whose thresholds stay the best.
    # Epoch 2 has the best F1 of all, but it comes after the selected epoch and is discarded with it.
    ([0.3, 0.5, 0.4], [0.2, 0.1, 0.9], 0),
    ([0.5, 0.4, 0.3], [0.1, 0.5, 0.9], 0),
    ([0.1, 0.2, 0.3], [0.1, 0.5, 0.9], 2),
    ([0.1, 0.2, 0.3], [0.1, 0.9, 0.5], 1),
])
def test_train_round_freezes_the_running_best_thresholds_at_the_selected_epoch(
    monkeypatch, val_maps, epoch_f1s, threshold_epoch
):
    epoch_thresholds = [np.full(3, 0.1 * (epoch + 1), dtype=np.float32) for epoch in range(len(val_maps))]
    _drive_validation_maps(monkeypatch, val_maps, epoch_thresholds=epoch_thresholds, epoch_f1s=epoch_f1s)
    _, round_result = train_round(**_train_kwargs(max_epochs=len(val_maps)))
    assert np.array_equal(round_result.thresholds, epoch_thresholds[threshold_epoch])
