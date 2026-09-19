"""Test the query contract of every registered strategy and the registry itself."""

from __future__ import annotations

import numpy as np
import pytest

import src.strategies.random_sampling as random_sampling_module
from src.strategies import (
    COLD_START_STRATEGY,
    FULL_SUPERVISED,
    STRATEGY_NAMES,
    QueryContext,
    QueryStrategy,
    build_query_strategy,
)
from src.strategies.auxiliary import nearest_labeled_targets
from src.strategies.backends import seed_backend
from src.strategies.badge import BADGEGreedyDPP, BADGEKMeansPlusPlus, BADGEMCMCDPP
from src.strategies.disagreement import Disagreement
from src.strategies.entropy import Entropy
from src.strategies.farthest_traversal import FarthestTraversal
from src.strategies.mfft import MFFT
from src.strategies.random_sampling import RandomSampling

_EXPECTED_CLASSES = {
    "random": RandomSampling,
    "entropy": Entropy,
    "farthest-traversal": FarthestTraversal,
    "disagreement": Disagreement,
    "mfft": MFFT,
    "badge-kmeans++": BADGEKMeansPlusPlus,
    "badge-mcmc-dpp": BADGEMCMCDPP,
    "badge-greedy-dpp": BADGEGreedyDPP,
}


def _build_properties(names, *, frame_embedding, frame_labels, rng, hidden=5):
    """Materialize exactly the declared properties, mirroring ActiveLearner._prepare_properties."""
    n, num_frames, num_classes = frame_labels.shape
    properties = {}
    if "seg_embedding" in names:
        # The cached segment representation is the mean over the frames of a segment.
        properties["seg_embedding"] = frame_embedding.mean(axis=1)
    if {"prediction_soft", "prediction_hard"} & set(names):
        proba = rng.random((n, num_frames, num_classes)).astype(np.float32)
        if "prediction_soft" in names:
            properties["prediction_soft"] = proba
        if "prediction_hard" in names:
            properties["prediction_hard"] = (proba >= 0.5).astype(np.int8)
    if "gradient" in names:
        properties["gradient"] = rng.standard_normal((n, num_classes, hidden)).astype(np.float32)
    return properties


def _required(name) -> tuple[str, ...]:
    """Return the property names the named strategy declares."""
    return _EXPECTED_CLASSES[name].required_properties


def _context(names=(), *, n=40, num_frames=6, num_classes=3, dim=8, num_labeled=8, step_size=6, seed=0):
    """Build a QueryContext holding exactly the named properties."""
    rng = np.random.default_rng(seed)
    frame_embedding = rng.standard_normal((n, num_frames, dim)).astype(np.float32)
    frame_labels = (rng.random((n, num_frames, num_classes)) > 0.75).astype(np.float32)
    labeled = np.arange(num_labeled, dtype=np.int64)
    return QueryContext(
        labeled=labeled,
        unlabeled=np.setdiff1d(np.arange(n, dtype=np.int64), labeled),
        step_size=step_size,
        n_train=n,
        frame_embedding=frame_embedding,
        frame_labels=frame_labels,
        xp=np,
        properties=_build_properties(names, frame_embedding=frame_embedding, frame_labels=frame_labels, rng=rng),
    )


def test_registry_lists_the_strategies_of_the_paper_in_order():
    assert STRATEGY_NAMES == (
        "random", "entropy", "farthest-traversal", "disagreement", "mfft",
        "badge-kmeans++", "badge-mcmc-dpp", "badge-greedy-dpp",
    )
    assert STRATEGY_NAMES == tuple(_EXPECTED_CLASSES)


def test_cold_start_and_reference_names():
    assert COLD_START_STRATEGY == "random" and COLD_START_STRATEGY in STRATEGY_NAMES
    assert FULL_SUPERVISED == "full-supervised" and FULL_SUPERVISED not in STRATEGY_NAMES


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_registry_builds_every_name(name):
    strategy = build_query_strategy(name, random_state=3)
    assert type(strategy) is _EXPECTED_CLASSES[name] and strategy.random_state == 3
    # The learner hands over its RandomState, which must be kept as is for the rounds to share one stream.
    shared = np.random.RandomState(3)
    assert build_query_strategy(name, random_state=shared).random_state is shared
    assert set(strategy.required_properties) <= {"seg_embedding", "prediction_soft", "prediction_hard", "gradient"}


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_strategy_selects_distinct_unlabeled_indices(name):
    ctx = _context(_required(name))
    seed_backend(5)
    selected = build_query_strategy(name, random_state=5).query(ctx)
    assert selected.dtype == np.int64 and selected.shape == (ctx.step_size,)
    assert len(set(selected.tolist())) == ctx.step_size
    assert set(selected.tolist()) <= set(ctx.unlabeled.tolist())


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_strategy_is_reproducible_for_the_same_seed(name):
    # The learner seeds the global streams once per experiment, so the kernels reading them repeat as well.
    first_ctx, second_ctx = _context(_required(name)), _context(_required(name))
    seed_backend(5)
    first = build_query_strategy(name, random_state=5).query(first_ctx)
    seed_backend(5)
    second = build_query_strategy(name, random_state=5).query(second_ctx)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", [n for n in STRATEGY_NAMES if _required(n)])
def test_strategy_raises_when_a_declared_property_is_missing(name):
    for missing in _required(name):
        ctx = _context(_required(name))
        del ctx.properties[missing]
        with pytest.raises(KeyError, match=missing):
            build_query_strategy(name, random_state=0).query(ctx)


def test_random_sampling_depends_on_the_seed():
    ctx = _context()
    seed_backend(3)
    first = RandomSampling(random_state=3).query(ctx)
    seed_backend(4)
    assert not np.array_equal(first, RandomSampling(random_state=4).query(ctx))


def test_random_sampling_accepts_a_seed_or_the_equivalent_random_state():
    ctx = _context()
    from_seed = RandomSampling(random_state=3).query(ctx)
    assert np.array_equal(from_seed, RandomSampling(random_state=np.random.RandomState(3)).query(ctx))


def test_random_sampling_draws_anew_as_the_labeled_pool_grows():
    # skactiveml seeds each query from a copy of the RandomState and the number of unlabeled segments, so
    # the rounds differ although the shared stream, which k-means++ also reads, is left where it was.
    shared = np.random.RandomState(3)
    first = RandomSampling(random_state=shared).query(_context(n=200, num_labeled=8))
    second = RandomSampling(random_state=shared).query(_context(n=200, num_labeled=14))
    assert sorted(first.tolist()) != sorted(second.tolist())
    assert shared.randint(2**31) == np.random.RandomState(3).randint(2**31)


def test_random_sampling_takes_the_whole_pool_without_consulting_skactiveml(monkeypatch):
    # The full-supervised reference labels every candidate at once; skactiveml would build a
    # [batch_size, n_train] utility matrix for a selection that involves no choice.
    def fail_if_called(*args, **kwargs):
        raise AssertionError("taking every candidate needs no sampler")

    monkeypatch.setattr(random_sampling_module, "SkactivemlRandomSampling", fail_if_called)
    ctx = _context(n=40, num_labeled=8, step_size=32)
    assert RandomSampling(random_state=0).query(ctx).tolist() == ctx.unlabeled.tolist()


def test_badge_kmeans_continues_a_shared_random_state_across_rounds():
    ctx = _context(("gradient",))
    shared = np.random.RandomState(5)
    first = BADGEKMeansPlusPlus(random_state=shared).query(ctx)
    second = BADGEKMeansPlusPlus(random_state=shared).query(ctx)
    # An integer seed starts a fresh stream and replays the first round; the shared stream has moved on.
    assert np.array_equal(first, BADGEKMeansPlusPlus(random_state=5).query(ctx))
    assert not np.array_equal(first, second)


def test_badge_selects_among_the_unlabeled_rows_of_a_pool_wide_gradient():
    # The gradient property covers the whole pool; a labeled segment with a huge gradient must stay out.
    ctx = _context(("gradient",))
    ctx.properties["gradient"][ctx.labeled] *= 1e3
    for name in ("badge-kmeans++", "badge-mcmc-dpp", "badge-greedy-dpp"):
        seed_backend(0)
        selected = build_query_strategy(name, random_state=0).query(ctx)
        assert set(selected.tolist()) <= set(ctx.unlabeled.tolist())


def test_entropy_takes_the_most_uncertain_segments():
    ctx = _context(("prediction_soft",), n=20, num_labeled=4, step_size=3)
    proba = ctx.properties["prediction_soft"]
    # Binary entropy grows towards 0.5, so taking the most uncertain class of the most uncertain frame
    # ranks the segments by the smallest distance from 0.5 anywhere in them.
    distance = np.abs(proba[ctx.unlabeled] - 0.5).min(axis=-1).min(axis=-1)
    expected = ctx.unlabeled[np.argsort(distance, kind="stable")[:3]]
    assert sorted(Entropy().query(ctx).tolist()) == sorted(expected.tolist())


def test_farthest_traversal_starts_from_the_labeled_set():
    from src.strategies.traversal import farthest_traversal

    ctx = _context(("seg_embedding",))
    _, expected = farthest_traversal(
        ctx.properties["seg_embedding"], ctx.step_size, ctx.labeled, ctx.unlabeled, np
    )
    assert FarthestTraversal().query(ctx).tolist() == expected.tolist()


def test_disagreement_prefers_the_segments_the_auxiliary_classifier_contradicts():
    from src.strategies.disagreement import mismatch_scores

    ctx = _context(("prediction_hard",))
    scores = mismatch_scores(ctx)
    selected = Disagreement().query(ctx)
    threshold = np.sort(scores)[-ctx.step_size]
    positions = {int(np.flatnonzero(ctx.unlabeled == idx)[0]) for idx in selected}
    assert all(scores[pos] >= threshold for pos in positions)


def test_nearest_labeled_targets_run_in_scikit_learn_for_numpy():
    # With numpy as the array module the search runs in scikit-learn, so that --device cpu stays off the GPU.
    rng = np.random.default_rng(0)
    labeled_features = rng.standard_normal((12, 5)).astype(np.float32)
    labeled_targets = (rng.random((12, 3)) > 0.5).astype(np.float32)
    query_frames = rng.standard_normal((7, 4, 5)).astype(np.float32)
    query_rows = np.array([5, 0, 3])
    reference = nearest_labeled_targets(labeled_features, labeled_targets, query_frames, query_rows, np)

    flat = query_frames[query_rows].reshape(-1, 5)
    nearest = np.linalg.norm(flat[:, None, :] - labeled_features[None, :, :], axis=-1).argmin(axis=1)
    assert reference.dtype == np.int8 and reference.shape == (3, 4, 3)
    assert np.array_equal(reference, labeled_targets[nearest].astype(np.int8).reshape(3, 4, 3))


def test_badge_variants_read_the_same_gradient_property():
    ctx = _context(("gradient",))
    assert {name for name in STRATEGY_NAMES if name.startswith("badge-")} == {
        "badge-kmeans++", "badge-mcmc-dpp", "badge-greedy-dpp"
    }
    selections = {}
    for name in ("badge-kmeans++", "badge-mcmc-dpp", "badge-greedy-dpp"):
        seed_backend(0)
        selections[name] = build_query_strategy(name, random_state=0).query(ctx).tolist()
        assert len(set(selections[name])) == ctx.step_size
    assert len({tuple(sorted(value)) for value in selections.values()}) == 3


class _TakeFirst(QueryStrategy):
    def _select(self, ctx):
        positions = ctx.unlabeled[:ctx.step_size]
        return np.arange(ctx.step_size, dtype=np.float32), positions


def test_base_query_orders_the_batch_by_score():
    ctx = _context(n=20, num_labeled=4, step_size=3)
    # Scores 0, 1, 2 go with candidates 4, 5, 6, and a larger score is better.
    assert _TakeFirst().query(ctx).tolist() == [6, 5, 4]


def test_base_normalize_breaks_ties_by_the_smaller_index():
    scores, idxes = QueryStrategy().normalize(np.array([1.0, 1.0, 2.0]), np.array([9, 3, 7]))
    assert idxes.tolist() == [7, 3, 9] and scores.tolist() == [2.0, 1.0, 1.0]
    assert scores.dtype == np.float32 and idxes.dtype == np.int64


def test_base_normalize_ranks_nan_scores_first():
    scores, idxes = QueryStrategy().normalize(np.array([1.0, np.nan, 2.0]), np.array([0, 1, 2]))
    assert idxes.tolist() == [1, 2, 0] and scores[0] > 2.0
