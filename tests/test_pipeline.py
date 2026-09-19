"""Test serial and multiprocessing pipelines with small CPU experiments."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import build_synthetic_frame_cache
from src.strategies import FULL_SUPERVISED, STRATEGY_NAMES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pipeline.py"
DATASET = "HyenaSET"

# The synthetic training pool has 60 segments, enough for two rounds selecting 6 each.
MAX_ITER = 2
STEP_SIZE = 6
SMALL_ARGS = [
    "--max-iter", str(MAX_ITER), "--step-size", str(STEP_SIZE),
    "--hidden-features", "16", "--max-epochs", "2", "--batch-size", "4",
    "--infer-batch-size", "8", "--device", "cpu",
]


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location("pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_root(tmp_path) -> Path:
    """Build a synthetic cache at tmp_path/cache/HyenaSET and return the cache root directory."""
    build_synthetic_frame_cache(tmp_path / "cache" / DATASET)
    return tmp_path / "cache"


def _root_args(cache_root: Path, output_root: Path) -> list[str]:
    return ["--cache-root", str(cache_root), "--output-root", str(output_root)]


def _results_csv(output_root: Path, strategy: str, seed: int) -> Path:
    (cfg_dir,) = (output_root / DATASET / strategy).iterdir()
    return cfg_dir / f"seed_{seed}" / "results.csv"


def _read_rows(results_csv: Path, *, drop: tuple[str, ...] = ()) -> list[dict]:
    """Read a results table, optionally without the columns that legitimately differ between runs."""
    with results_csv.open(newline="") as fh:
        return [{key: value for key, value in row.items() if key not in drop} for row in csv.DictReader(fh)]


def test_parse_args_defaults_follow_paper_protocol(pipeline, monkeypatch):
    monkeypatch.setenv("CACHE_ROOT", "/cache-from-env")
    monkeypatch.setenv("OUTPUT_ROOT", "/output-from-env")
    args = pipeline.parse_args([])
    assert args.datasets == [DATASET]
    assert args.strategies == list(STRATEGY_NAMES)
    assert (args.seed, args.runs, args.max_iter, args.step_size) == (0, 10, 10, 300)
    assert (args.hidden_features, args.lr, args.max_epochs) == (0, 1e-3, 1000)
    assert (args.batch_size, args.infer_batch_size) == (32, 1024)
    assert (args.device, args.n_tasks) == ("auto", 4)
    assert (args.cache_root, args.output_root) == ("/cache-from-env", "/output-from-env")


def test_parse_args_accepts_the_full_supervised_reference(pipeline):
    assert pipeline.parse_args(["--strategies", FULL_SUPERVISED]).strategies == [FULL_SUPERVISED]


@pytest.mark.parametrize("argv", [
    ["--strategies", "badge"],
    ["--datasets", "DESED"],
    ["--device", "cuda:0"],
])
def test_parse_args_rejects_unknown_choices(pipeline, argv):
    with pytest.raises(SystemExit):
        pipeline.parse_args(argv)


def test_main_requires_roots(pipeline, monkeypatch):
    monkeypatch.delenv("CACHE_ROOT", raising=False)
    monkeypatch.delenv("OUTPUT_ROOT", raising=False)
    with pytest.raises(SystemExit):
        pipeline.main([])


def test_prepare_tasks_fails_fast_on_a_missing_cache(pipeline, tmp_path):
    args = pipeline.parse_args([*SMALL_ARGS, *_root_args(tmp_path / "absent", tmp_path / "out")])
    with pytest.raises(SystemExit):
        pipeline.prepare_tasks(args, tmp_path / "absent", tmp_path / "out", "cpu")


def test_prepare_tasks_expands_strategies_and_seeds(pipeline, cache_root, tmp_path):
    output_root = tmp_path / "out"
    argv = ["--strategies", "random", "mfft", "--runs", "3", *SMALL_ARGS,
            *_root_args(cache_root, output_root)]
    tasks, n_skipped = pipeline.prepare_tasks(pipeline.parse_args(argv), cache_root, output_root, "cpu")
    assert n_skipped == 0
    assert len(tasks) == 2 * 3
    assert {task["learner"]["strategy_name"] for task in tasks} == {"random", "mfft"}
    assert sorted(task["learner"]["seed"] for task in tasks) == [0, 0, 1, 1, 2, 2]
    assert all(task["learner"]["step_size"] == STEP_SIZE for task in tasks)


def test_prepare_tasks_counts_the_seeds_up_from_the_base_seed(pipeline, cache_root, tmp_path):
    output_root = tmp_path / "out"
    argv = ["--strategies", "random", "--seed", "5", "--runs", "2", *SMALL_ARGS,
            *_root_args(cache_root, output_root)]
    tasks, _ = pipeline.prepare_tasks(pipeline.parse_args(argv), cache_root, output_root, "cpu")
    assert [task["learner"]["seed"] for task in tasks] == [5, 6]
    # Seeds of one configuration share its directory, so the seed is not part of config.json.
    (config_path,) = (output_root / DATASET / "random").glob("*/config.json")
    assert "seed" not in json.loads(config_path.read_text())


def test_prepare_tasks_gives_the_full_supervised_reference_the_whole_pool(pipeline, cache_root, tmp_path):
    output_root = tmp_path / "out"
    argv = ["--strategies", FULL_SUPERVISED, "random", "--runs", "1", *SMALL_ARGS,
            *_root_args(cache_root, output_root)]
    tasks, _ = pipeline.prepare_tasks(pipeline.parse_args(argv), cache_root, output_root, "cpu")
    reference = next(t["learner"] for t in tasks if t["learner"]["strategy_name"] == FULL_SUPERVISED)
    active = next(t["learner"] for t in tasks if t["learner"]["strategy_name"] == "random")
    assert (reference["step_size"], reference["max_iter"]) == (60, 1)
    assert (active["step_size"], active["max_iter"]) == (STEP_SIZE, MAX_ITER)
    # A different step size and round count must land in a different configuration directory.
    config = json.loads((output_root / DATASET / FULL_SUPERVISED).glob("*/config.json").__next__().read_text())
    assert config["strategy"] == FULL_SUPERVISED and config["n_train"] == 60


def test_prepare_tasks_skips_only_complete_seeds(pipeline, cache_root, tmp_path):
    output_root = tmp_path / "out"
    argv = ["--strategies", "random", "--runs", "2", "--n-tasks", "1", *SMALL_ARGS,
            *_root_args(cache_root, output_root)]
    pipeline.main(argv)

    truncated = _results_csv(output_root, "random", seed=1)
    lines = truncated.read_text().splitlines()
    truncated.write_text("\n".join(lines[:-1]) + "\n")

    tasks, n_skipped = pipeline.prepare_tasks(pipeline.parse_args(argv), cache_root, output_root, "cpu")
    assert n_skipped == 1
    assert [task["learner"]["seed"] for task in tasks] == [1]


def test_main_serial_end_to_end(pipeline, cache_root, tmp_path, monkeypatch, capsys):
    output_root = tmp_path / "out"
    argv = ["--strategies", "random", "badge-greedy-dpp", "--runs", "1", "--n-tasks", "1",
            *SMALL_ARGS, *_root_args(cache_root, output_root)]
    pipeline.main(argv)

    for strategy in ("random", "badge-greedy-dpp"):
        results_csv = _results_csv(output_root, strategy, seed=0)
        assert pipeline.results_csv_complete(results_csv, MAX_ITER)
        rows = _read_rows(results_csv)
        assert [int(row["num_labeled_samples"]) for row in rows] == [STEP_SIZE, 2 * STEP_SIZE]
        assert all(float(row["query_time"]) >= 0.0 for row in rows)
        selected = [json.loads(row["queried_idxes_in_latest_iteration"]) for row in rows]
        assert all(len(batch) == STEP_SIZE for batch in selected)
        assert len(set(selected[0]) | set(selected[1])) == MAX_ITER * STEP_SIZE
        # Round weights are not kept, so the table is the only file of a seed.
        assert [p.name for p in results_csv.parent.iterdir()] == ["results.csv"]

    def fail_if_constructed(**kwargs):
        raise AssertionError("a completed seed must not run again")

    monkeypatch.setattr(pipeline, "ActiveLearner", fail_if_constructed)
    capsys.readouterr()
    pipeline.main(argv)
    assert "2 completed and skipped, 0 to run" in capsys.readouterr().out


def test_round_zero_is_shared_across_strategies(pipeline, cache_root, tmp_path):
    """Every strategy must start from the same cold-start set, which is what makes the comparison paired."""
    output_root = tmp_path / "out"
    pipeline.main(["--strategies", *STRATEGY_NAMES, "--runs", "1", "--n-tasks", "1",
                   *SMALL_ARGS, *_root_args(cache_root, output_root)])
    first_rounds = {
        strategy: _read_rows(_results_csv(output_root, strategy, seed=0))[0]["queried_idxes_in_latest_iteration"]
        for strategy in STRATEGY_NAMES
    }
    assert len(first_rounds) == len(STRATEGY_NAMES) and len(set(first_rounds.values())) == 1


def _seed_files(output_root: Path) -> list[Path]:
    """Return sorted relative file paths within each seed directory."""
    return sorted(p.relative_to(output_root) for p in output_root.rglob("seed_*/*") if p.is_file())


def test_parallel_matches_serial(cache_root, tmp_path):
    # k-means++ draws from the per-run RandomState and the MCMC DPP from the global stream seeded per run.
    common = ["--strategies", "entropy", "badge-kmeans++", "badge-mcmc-dpp", "--runs", "2",
              *SMALL_ARGS, "--cache-root", str(cache_root)]
    # Subprocesses launch as scripts without inheriting OMP_NUM_THREADS, matching the actual runtime environment.
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    for mode, n_tasks in (("serial", "1"), ("parallel", "2")):
        subprocess.run(
            [sys.executable, str(SCRIPT), *common, "--n-tasks", n_tasks,
             "--output-root", str(tmp_path / mode)],
            check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    serial_root, parallel_root = tmp_path / "serial", tmp_path / "parallel"
    relative_paths = _seed_files(serial_root)
    assert relative_paths == _seed_files(parallel_root)
    assert len(relative_paths) == 3 * 2  # Strategies x seeds, each seed directory holding only results.csv
    for relative in relative_paths:
        # Wall-clock query time is the one column that cannot match; everything else must, whether the
        # runs of a process share its random streams one after another or each get a process of their own.
        assert _read_rows(serial_root / relative, drop=("query_time",)) == _read_rows(
            parallel_root / relative, drop=("query_time",)
        ), relative
