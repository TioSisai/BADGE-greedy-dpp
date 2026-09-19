"""Test the array backend, the BADGE gradient, and the traversal, k-means++ and DPP kernels on numpy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import src.strategies.dpp as dpp_module
from src.strategies import backends
from src.strategies.backends import (
    array_module,
    backend_signature,
    pairwise_distances,
    require_backend,
    seed_backend,
    solve_lower_triangular,
    to_backend,
    to_numpy,
)
from src.strategies.badge import badge_gradient
from src.strategies.dpp import LAMBDA_REG, greedy_dpp, mcmc_dpp
from src.strategies.kmeans_plusplus import kmeans_plusplus
from src.strategies.traversal import farthest_traversal


def _cluster_features(num_clusters: int = 4, per_cluster: int = 3, scale: float = 5.0) -> np.ndarray:
    """Return near-duplicate clusters, each spread around its own basis direction."""
    rng = np.random.default_rng(0)
    features = np.zeros((num_clusters * per_cluster, num_clusters), dtype=np.float64)
    for cluster in range(num_clusters):
        rows = slice(cluster * per_cluster, (cluster + 1) * per_cluster)
        features[rows, cluster] = scale
    return features + 1e-4 * rng.standard_normal(features.shape)


def test_backend_signature_follows_the_device_and_the_gpu_backend(monkeypatch):
    assert backend_signature("cpu") == "sklearn+numpy"
    # The CPU fallback: a CUDA device is served by numpy and scikit-learn when the GPU backend is not usable.
    monkeypatch.setattr(backends, "HAS_GPU_BACKEND", False)
    assert backend_signature("cuda") == "sklearn+numpy"


def test_require_backend_rejects_a_signature_this_process_cannot_provide():
    require_backend("sklearn+numpy", "cpu")
    with pytest.raises(RuntimeError, match="requires cuml\\+cupy"):
        require_backend("cuml+cupy", "cpu")


def test_array_module_is_numpy_on_cpu_and_without_the_gpu_backend(monkeypatch):
    assert array_module("cpu") is np
    monkeypatch.setattr(backends, "HAS_GPU_BACKEND", False)
    assert array_module("cuda") is np and array_module("cuda:0") is np


def test_to_numpy_and_to_backend_round_trip():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert to_numpy(array) is array
    assert to_backend(array, np) is array  # A numpy input passes through without a copy or a dtype change.

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert np.array_equal(to_numpy(tensor), array)
    assert np.array_equal(to_backend(tensor, np), array)


def test_pairwise_distances_values():
    a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    distances = pairwise_distances(a, b)
    assert distances.shape == (2, 2)
    assert np.allclose(to_numpy(distances), [[0.0, 2.0], [1.0, np.sqrt(5.0)]], atol=1e-5)
    assert np.allclose(to_numpy(pairwise_distances(a, a)), [[0.0, 1.0], [1.0, 0.0]], atol=1e-5)


def test_solve_lower_triangular_matches_the_dense_solution():
    lower = np.array([[2.0, 0.0], [1.0, 3.0]])
    rhs = np.array([[4.0, 2.0], [7.0, 6.0]])
    assert np.allclose(to_numpy(solve_lower_triangular(lower, rhs, np)), np.linalg.solve(lower, rhs))


def test_seed_backend_makes_the_global_numpy_stream_reproducible():
    seed_backend(7)
    first = np.random.random(4)
    seed_backend(7)
    assert np.array_equal(first, np.random.random(4))


def test_farthest_traversal_takes_the_candidate_farthest_from_the_anchors():
    # The anchor sits at the origin, so candidate 3 is taken first and candidate 2 second.
    features = np.array([[0.0], [1.0], [2.0], [5.0]], dtype=np.float64)
    scores, idxes = farthest_traversal(features, 2, np.array([0]), np.array([1, 2, 3]), np)
    assert to_numpy(idxes).tolist() == [3, 2]
    assert np.allclose(to_numpy(scores), [5.0, 2.0])


def test_farthest_traversal_is_deterministic_and_distinct():
    features = np.random.default_rng(0).standard_normal((30, 4))
    anchors, candidates = np.array([0, 1]), np.arange(2, 30)
    first = to_numpy(farthest_traversal(features, 8, anchors, candidates, np)[1]).tolist()
    second = to_numpy(farthest_traversal(features, 8, anchors, candidates, np)[1]).tolist()
    assert first == second and len(set(first)) == 8 and set(first) <= set(candidates.tolist())


def test_badge_gradient_matches_a_hand_computed_case():
    # The residual of frame t and class c is p - [p > 0.5], and the gradient averages its outer product
    # with the penultimate feature over the frames.
    proba = np.array([[[0.8, 0.2], [0.3, 0.9]]], dtype=np.float32)
    last_feature = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
    gradient = badge_gradient(proba, last_feature, np)
    assert gradient.shape == (1, 2, 3) and gradient.dtype == np.float32
    assert np.allclose(gradient[0], [[0.5, 0.55, 0.6], [-0.1, -0.05, 0.0]], atol=1e-6)


def test_badge_gradient_accepts_torch_tensors():
    proba = torch.tensor([[[0.8, 0.2], [0.3, 0.9]]])
    last_feature = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    gradient = badge_gradient(proba, last_feature, np)
    assert isinstance(gradient, np.ndarray)
    assert np.allclose(gradient[0], [[0.5, 0.55, 0.6], [-0.1, -0.05, 0.0]], atol=1e-6)


def test_badge_gradient_is_zero_for_a_confident_prediction():
    proba = np.full((2, 4, 3), 1.0, dtype=np.float32)
    last_feature = np.ones((2, 4, 5), dtype=np.float32)
    assert np.allclose(badge_gradient(proba, last_feature, np), 0.0)


def test_kmeans_plusplus_is_reproducible_for_a_fixed_random_state():
    features = _cluster_features().astype(np.float32)
    scores, idxes = kmeans_plusplus(features, 4, np, random_state=3)
    again_scores, again_idxes = kmeans_plusplus(features, 4, np, random_state=3)
    assert to_numpy(idxes).tolist() == to_numpy(again_idxes).tolist()
    assert np.array_equal(to_numpy(scores)[1:], to_numpy(again_scores)[1:])


def test_kmeans_plusplus_accepts_a_seed_or_the_equivalent_random_state():
    features = _cluster_features().astype(np.float32)
    _, from_seed = kmeans_plusplus(features, 4, np, random_state=3)
    _, from_state = kmeans_plusplus(features, 4, np, random_state=np.random.RandomState(3))
    assert to_numpy(from_seed).tolist() == to_numpy(from_state).tolist()


def test_kmeans_plusplus_continues_a_shared_random_state_instead_of_replaying_it():
    features = np.random.default_rng(0).standard_normal((40, 4)).astype(np.float32)
    shared = np.random.RandomState(3)
    first = to_numpy(kmeans_plusplus(features, 6, np, random_state=shared)[1]).tolist()
    second = to_numpy(kmeans_plusplus(features, 6, np, random_state=shared)[1]).tolist()
    assert first == to_numpy(kmeans_plusplus(features, 6, np, random_state=3)[1]).tolist()
    assert second != first

    # One call draws the first seed once and then 2 + int(ln n) local trials for each of the other seeds,
    # so a fresh stream advanced by exactly these draws reproduces the second call.
    advanced = np.random.RandomState(3)
    advanced.choice(40, p=np.full(40, 1.0 / 40))
    for _ in range(5):
        advanced.uniform(size=2 + int(np.log(6)))
    assert second == to_numpy(kmeans_plusplus(features, 6, np, random_state=advanced)[1]).tolist()


def test_kmeans_plusplus_matches_scikit_learn_for_the_same_random_state():
    from sklearn.cluster import kmeans_plusplus as sklearn_kmeans_plusplus

    features = np.random.default_rng(0).standard_normal((60, 5))
    for seed in range(3):
        _, expected = sklearn_kmeans_plusplus(features, 8, random_state=np.random.RandomState(seed))
        _, idxes = kmeans_plusplus(features, 8, np, random_state=np.random.RandomState(seed))
        assert to_numpy(idxes).tolist() == expected.tolist()


def test_kmeans_plusplus_first_score_is_nan_and_the_rest_are_finite():
    features = _cluster_features().astype(np.float32)
    scores, _ = kmeans_plusplus(features, 4, np, random_state=0)
    scores = to_numpy(scores)
    assert np.isnan(scores[0]) and np.isfinite(scores[1:]).all()


def test_kmeans_plusplus_spreads_over_the_clusters():
    features = _cluster_features().astype(np.float32)
    _, idxes = kmeans_plusplus(features, 4, np, random_state=1)
    selected = to_numpy(idxes).tolist()
    assert len(set(selected)) == 4 and len({pos // 3 for pos in selected}) == 4


def test_greedy_dpp_is_deterministic_and_returns_distinct_indices():
    features = np.random.default_rng(0).standard_normal((20, 6))
    scores, idxes = greedy_dpp(features, 5, np)
    again_scores, again_idxes = greedy_dpp(features, 5, np)
    selected = to_numpy(idxes).tolist()
    assert selected == to_numpy(again_idxes).tolist()
    assert np.allclose(to_numpy(scores), to_numpy(again_scores))
    assert len(set(selected)) == 5 and set(selected) <= set(range(20))


def test_greedy_dpp_first_pick_is_the_largest_marginal_volume():
    # The empty selection has kernel LAMBDA_REG * I, so the first gain is 1 + ||phi||^2 / LAMBDA_REG.
    features = np.array([[1.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
    scores, idxes = greedy_dpp(features, 1, np)
    assert int(to_numpy(idxes)[0]) == 1
    assert float(to_numpy(scores)[0]) == pytest.approx(1.0 + 9.0 / LAMBDA_REG, rel=1e-9)


def test_greedy_dpp_visits_every_cluster_before_revisiting_one():
    features = _cluster_features(num_clusters=4, per_cluster=3)
    _, idxes = greedy_dpp(features, 6, np)
    selected = to_numpy(idxes).tolist()
    assert len({pos // 3 for pos in selected[:4]}) == 4  # One representative per near-duplicate cluster
    assert len(set(selected)) == 6


def _stepwise_greedy_dpp(X, n):
    """Rescore the whole pool at every step and take the argmax, which is what the cupy branch does."""
    from scipy.linalg import solve_triangular

    current_l = np.sqrt(LAMBDA_REG) * np.eye(X.shape[1])
    available = np.ones(X.shape[0], dtype=bool)
    scores, positions = np.empty(n), np.empty(n, dtype=np.int64)
    for i in range(n):
        solved = solve_triangular(current_l, X.T, lower=True)
        gains = np.where(available, (solved * solved).sum(axis=0) + 1.0, -np.inf)
        positions[i] = gains.argmax()
        scores[i] = gains[positions[i]]
        available[positions[i]] = False
        current_l = np.linalg.qr(np.vstack((current_l.T, X[positions[i]][None, :])), mode="r").T
    return scores, positions


def _badge_like_gradients(num_rows=400, num_frames=6, num_classes=4, hidden=8, seed=0):
    """Return residual-times-feature embeddings, most of them tiny because the predictions are confident."""
    rng = np.random.default_rng(seed)
    proba = rng.beta(0.2, 0.2, size=(num_rows, num_frames, num_classes))
    residual = proba - (proba > 0.5)
    feature = rng.standard_normal((num_rows, num_frames, hidden))
    return np.einsum("ntc,ntd->ncd", residual, feature).reshape(num_rows, -1) / num_frames


def _exact_duplicate_features(num_base=40, num_duplicates=20, dim=8, seed=3):
    """Return Gaussian rows and exact copies of some of them in shuffled order, so their gains tie bit for bit."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((num_base, dim))
    features = np.vstack((base, base[:num_duplicates]))
    return features[rng.permutation(features.shape[0])]


@pytest.mark.parametrize("features,n", [
    (np.random.default_rng(0).standard_normal((300, 24)), 40),
    (np.random.default_rng(1).standard_normal((80, 6)), 30),  # More picks than dimensions
    (_cluster_features(num_clusters=6, per_cluster=5), 12),
    (_badge_like_gradients(), 60),
    # Both copies of a row reach the top with the same gain, which argmax resolves by the smaller position.
    (_exact_duplicate_features(), 12),
], ids=["gaussian", "rank-saturated", "near-duplicates", "badge-like", "exact-duplicates"])
def test_greedy_dpp_lazy_branch_matches_the_stepwise_argmax(features, n):
    scores, positions = greedy_dpp(features, n, np)
    expected_scores, expected_positions = _stepwise_greedy_dpp(features, n)
    assert positions.tolist() == expected_positions.tolist()
    assert np.allclose(scores, expected_scores, rtol=1e-9, atol=0.0)


def test_greedy_dpp_takes_the_smaller_position_of_an_exactly_tied_pair():
    features = _exact_duplicate_features()
    _, positions = greedy_dpp(features, 12, np)
    step_of = {pos: step for step, pos in enumerate(positions.tolist())}
    num_rows = features.shape[0]
    twins = [
        (pos, twin) for pos in range(num_rows) for twin in range(pos + 1, num_rows)
        if np.array_equal(features[pos], features[twin])
    ]
    reached = [(pos, twin) for pos, twin in twins if pos in step_of or twin in step_of]
    assert reached  # The batch must run into tied pairs for this case to say anything.
    # Twins tie until one of them is taken, and that one must be the smaller position, as argmax would pick.
    assert all(pos in step_of and step_of[pos] < step_of.get(twin, len(step_of)) for pos, twin in reached)


def test_greedy_dpp_scores_the_whole_pool_only_once_on_numpy(monkeypatch):
    solved_columns = []
    original = dpp_module.solve_lower_triangular

    def recording(lower, rhs, xp):
        solved_columns.append(rhs.shape[1])
        return original(lower, rhs, xp)

    monkeypatch.setattr(dpp_module, "solve_lower_triangular", recording)
    greedy_dpp(np.random.default_rng(0).standard_normal((200, 10)), 20, np)
    # After the initial pass, only single candidates popped from the heap are rescored.
    assert solved_columns[0] == 200 and set(solved_columns[1:]) == {1}


def test_mcmc_dpp_returns_the_requested_number_of_distinct_indices():
    features = np.random.default_rng(1).standard_normal((25, 5))
    seed_backend(0)
    scores, idxes = mcmc_dpp(features, 6, np)
    selected = to_numpy(idxes).tolist()
    assert len(selected) == len(set(selected)) == 6 and set(selected) <= set(range(25))
    assert to_numpy(scores).shape == (6,) and (to_numpy(scores) == 1.0).all()


def test_mcmc_dpp_repeats_under_the_same_global_seed():
    # The chain draws from the global stream the learner seeds once per experiment.
    features = np.random.default_rng(1).standard_normal((25, 5))
    seed_backend(4)
    first = to_numpy(mcmc_dpp(features, 6, np)[1]).tolist()
    seed_backend(4)
    assert to_numpy(mcmc_dpp(features, 6, np)[1]).tolist() == first


def _boolean_mask_mcmc_dpp(X, n):
    """The chain as first ported, with boolean masks, kept to pin the integer-index version bit for bit."""
    xp = np

    def gram_red(current_kernel, current_kernel_inv, remove_pos):
        remove_pos_vector = xp.asarray(remove_pos, dtype=xp.int64).reshape(-1)
        keep_mask = xp.arange(current_kernel.shape[0]) != remove_pos
        reduced_kernel = current_kernel[keep_mask][:, keep_mask]
        reduced_kernel_inv = current_kernel_inv[keep_mask][:, keep_mask]
        column = current_kernel_inv[keep_mask][:, remove_pos_vector]
        diagonal = current_kernel_inv[remove_pos_vector][:, remove_pos_vector]
        return reduced_kernel, reduced_kernel_inv - (column @ column.T) / diagonal, keep_mask

    num_candidates = X.shape[0]
    subset_positions = xp.random.choice(num_candidates, size=n, replace=False).astype(xp.int64, copy=False)
    in_subset = xp.zeros(num_candidates, dtype=bool)
    in_subset[subset_positions] = True
    current_kernel = X[subset_positions] @ X[subset_positions].T
    current_kernel_inv = xp.linalg.pinv(current_kernel)
    for _ in range(1, int(5 * n * np.log(n))):
        outside_positions = xp.flatnonzero(~in_subset)
        remove_local_pos = xp.random.randint(0, subset_positions.shape[0])
        remove_candidate_pos = xp.asarray(subset_positions[remove_local_pos], dtype=xp.int64).reshape(-1)
        add_candidate_pos = xp.asarray(
            outside_positions[xp.random.randint(0, outside_positions.shape[0])], dtype=xp.int64
        ).reshape(-1)
        reduced_kernel, reduced_kernel_inv, keep_mask = gram_red(current_kernel, current_kernel_inv, remove_local_pos)
        reduced_features = X[subset_positions[keep_mask]]
        removed_gain = dpp_module._removal_gain(reduced_features, X[remove_candidate_pos], reduced_kernel_inv)
        added_feature = X[add_candidate_pos]
        augmented_kernel, augmented_kernel_inv, added_gain = dpp_module._gram_aug(
            reduced_kernel, reduced_kernel_inv, reduced_features @ added_feature.T,
            added_feature @ added_feature.T, xp,
        )
        accept = xp.asarray(
            xp.random.random() <= xp.minimum(1.0, added_gain / removed_gain).reshape(()), dtype=bool
        )
        updated_in_subset = in_subset.copy()
        updated_in_subset[remove_candidate_pos] = False
        updated_in_subset[add_candidate_pos] = True
        in_subset = xp.where(accept, updated_in_subset, in_subset)
        subset_positions = xp.where(
            accept, xp.concatenate((subset_positions[keep_mask], add_candidate_pos)), subset_positions
        )
        current_kernel = xp.where(accept, augmented_kernel, current_kernel)
        current_kernel_inv = xp.where(accept, augmented_kernel_inv, current_kernel_inv)
    return subset_positions


@pytest.mark.parametrize("num_rows,dim,n", [(60, 12, 8), (150, 20, 25), (90, 6, 12)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mcmc_dpp_integer_indexing_keeps_the_chain_bit_identical(num_rows, dim, n, seed):
    features = np.random.default_rng(seed).standard_normal((num_rows, dim))
    seed_backend(seed)
    expected = _boolean_mask_mcmc_dpp(features, n)
    expected_stream = np.random.get_state()
    seed_backend(seed)
    _, positions = mcmc_dpp(features, n, np)
    # The same subset in the same order, reached through the same sequence of random draws.
    assert positions.tolist() == expected.tolist()
    stream = np.random.get_state()
    assert np.array_equal(stream[1], expected_stream[1]) and stream[2:] == expected_stream[2:]
