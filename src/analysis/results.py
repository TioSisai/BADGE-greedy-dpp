"""Result discovery, curve loading, the paper summary table, and paired significance tests."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import permutation_test, wilcoxon

from ..data import DATASET_NAMES
from ..data.arrays import META_FILENAME

# Mirrors the strategy registry order without importing the strategy implementations.
STRATEGY_ORDER = (
    "random",
    "entropy",
    "farthest-traversal",
    "disagreement",
    "mfft",
    "badge-kmeans++",
    "badge-mcmc-dpp",
    "badge-greedy-dpp",
)
# Mirrors the registry constant. The reference labels the whole pool in one round and is not summarized as a strategy.
FULL_SUPERVISED = "full-supervised"
CURVE_STRATEGY_ORDER = STRATEGY_ORDER + (FULL_SUPERVISED,)
# Default reference strategy of the paired tests.
REFERENCE_STRATEGY = "badge-greedy-dpp"

DISPLAY_NAMES = {
    "random": "Random",
    "entropy": "Entropy",
    "farthest-traversal": "Farthest Traversal",
    "disagreement": "Disagreement",
    "mfft": "MFFT",
    "badge-kmeans++": "Vanilla BADGE (KMeans++)",
    "badge-mcmc-dpp": "Vanilla BADGE (MCMC DPP)",
    "badge-greedy-dpp": "BADGE Greedy DPP (proposal)",
    FULL_SUPERVISED: "Full supervised",
}

# Colors and markers shared by every figure.
STRATEGY_COLORS = {
    "random": "#808080",
    "entropy": "#1F77B4",
    "farthest-traversal": "#2CA02C",
    "disagreement": "#FF7F0E",
    "mfft": "#9467BD",
    "badge-kmeans++": "#8C564B",
    "badge-mcmc-dpp": "#17BECF",
    "badge-greedy-dpp": "#D62728",
}
STRATEGY_MARKERS = {
    "random": "o",
    "entropy": "s",
    "farthest-traversal": "D",
    "disagreement": "^",
    "mfft": "v",
    "badge-kmeans++": "P",
    "badge-mcmc-dpp": "X",
    "badge-greedy-dpp": "*",
}
FULL_SUPERVISED_COLOR = "#4D4D4D"

# Call types whose segment prevalence sits in the tail of the training pool.
RARE_CLASSES = ("gwl", "snr", "str")

# Mirror the output layout of src.learner without importing the training stack.
CONFIG_FILENAME = "config.json"
RESULTS_FILENAME = "results.csv"
SELECTION_COLUMN = "queried_idxes_in_latest_iteration"
CLASS_MAP_PREFIX = "test_mAP_"
# Segment-level targets of the training pool, the only cache file the summaries read.
SEG_LABELS_FILE = "train_seg_labels.npy"

CURVE_COLUMNS = (
    "strategy", "seed", "iteration", "num_labeled", "query_time", "test_mAP", "rare_mAP", "selection",
)
# Paper metric labels, used as column stems of summary_table and as the metric argument of the paired tests.
SEED_METRICS = ("N-AULC", "Rare-N-AULC", "F-mAP", "F-rmAP", "R-Enr", "FS-Bud", "QT")
SUMMARY_METRICS = ("N-AULC", "Rare-N-AULC", "F-mAP", "F-rmAP", "R-Enr", "FS-Bud", "FS-Rch", "QT")

# The sign-flip test enumerates every pattern up to this many pairs and samples random patterns beyond it.
EXACT_PERMUTATION_MAX_PAIRS = 20
PERMUTATION_RESAMPLES = 100_000
PERMUTATION_RNG_SEED = 0
# Sign patterns that are mathematically tied with the observed statistic may differ by a few ulp.
PERMUTATION_TIE_TOL = 1e-12


def parse_config_filter(items: Iterable[str]) -> dict:
    """Parse KEY=VALUE conditions into a dictionary, reading values as JSON when possible and as strings otherwise.

    Raises:
        ValueError: A condition is missing an equals sign.
    """
    config_filter = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"filter condition must be of the form KEY=VALUE: {item!r}")
        try:
            config_filter[key] = json.loads(raw)
        except json.JSONDecodeError:
            config_filter[key] = raw
    return config_filter


def _describe_candidates(cfg_dirs: Sequence[Path]) -> str:
    """List fields that differ among candidate configurations for the same strategy."""
    configs = [json.loads((d / CONFIG_FILENAME).read_text(encoding="utf-8")) for d in cfg_dirs]
    keys = sorted(set().union(*configs))
    differing = [k for k in keys if len({json.dumps(c.get(k)) for c in configs}) > 1]
    return "\n".join(
        f"    {d.name}: " + ", ".join(f"{k}={c.get(k)!r}" for k in differing)
        for d, c in zip(cfg_dirs, configs)
    )


def discover_config_dirs(
    output_root,
    dataset: str = DATASET_NAMES[0],
    strategies: Iterable[str] = STRATEGY_ORDER,
    *,
    config_hashes: Sequence[str] | None = None,
    config_filter: Mapping | None = None,
) -> dict[str, Path]:
    """Locate the unique configuration directory of each strategy for one dataset.

    Args:
        output_root: Root directory for experiment outputs.
        dataset: Dataset name.
        strategies: Sequence of strategy names, optionally including FULL_SUPERVISED.
        config_hashes: Optional prefixes of the configuration directory names to keep.
        config_filter: Optional conditions on configuration fields, all of which a configuration must satisfy.

    Returns:
        ``{strategy: config_dir}``, preserving the input order and omitting strategies without matches.

    Raises:
        ValueError: Multiple configurations remain after filtering. The message lists the differing fields.
    """
    prefixes = tuple(config_hashes or ())
    found: dict[str, Path] = {}
    ambiguous: list[str] = []
    for strategy in strategies:
        candidates = []
        for cfg_path in sorted((Path(output_root) / dataset / strategy).glob(f"*/{CONFIG_FILENAME}")):
            if prefixes and not cfg_path.parent.name.startswith(prefixes):
                continue
            if config_filter:
                config = json.loads(cfg_path.read_text(encoding="utf-8"))
                if any(config.get(k) != v for k, v in config_filter.items()):
                    continue
            candidates.append(cfg_path.parent)
        if len(candidates) == 1:
            found[strategy] = candidates[0]
        elif candidates:
            ambiguous.append(f"  {dataset}/{strategy}:\n{_describe_candidates(candidates)}")
    if ambiguous:
        raise ValueError(
            "these strategies map to multiple config_hash dirs, disambiguate with --config-hash or --config-filter:\n"
            + "\n".join(ambiguous)
        )
    return found


def seed_dirs(config_dir) -> dict[int, Path]:
    """Return ``{seed: seed_dir}`` under the configuration directory in ascending seed order."""
    runs = ((int(p.name.removeprefix("seed_")), p) for p in Path(config_dir).glob("seed_*"))
    return dict(sorted(runs))


def _finalize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Put the columns in CURVE_COLUMNS order, order strategies by CURVE_STRATEGY_ORDER, and sort the rows."""
    curves["strategy"] = pd.Categorical(curves["strategy"], categories=list(CURVE_STRATEGY_ORDER), ordered=True)
    extra = [column for column in curves.columns if column not in CURVE_COLUMNS]
    curves = curves[[*CURVE_COLUMNS, *extra]]
    # Row order fixes the order of later floating-point reductions, so it must not depend on the filesystem.
    return curves.sort_values(["strategy", "seed", "iteration"]).reset_index(drop=True)


def load_curves(config_dirs: Mapping[str, Path]) -> pd.DataFrame:
    """Collect per-round metrics for completed seeds, warning about and skipping incomplete seeds.

    Args:
        config_dirs: Return value of discover_config_dirs.

    Returns:
        A long-format table sorted by strategy, seed and round, with columns
        ``[strategy, seed, iteration, num_labeled, query_time, test_mAP, rare_mAP, selection]``.
        rare_mAP averages the RARE_CLASSES columns, selection holds the queried global train-pool
        indices of that round in selection order, and empty values are read as NaN.

    Raises:
        ValueError: No completed seeds are available.
    """
    rare_columns = [f"{CLASS_MAP_PREFIX}{name}" for name in RARE_CLASSES]
    frames = []
    skipped = []
    for strategy, cfg_dir in config_dirs.items():
        max_iter = json.loads((Path(cfg_dir) / CONFIG_FILENAME).read_text(encoding="utf-8"))["max_iter"]
        for seed, run_dir in seed_dirs(cfg_dir).items():
            df = pd.read_csv(
                run_dir / RESULTS_FILENAME,
                usecols=["iteration", "num_labeled_samples", "query_time", "test_mAP",
                         *rare_columns, SELECTION_COLUMN],
            )
            if len(df) != max_iter:
                skipped.append(str(run_dir))
                continue
            frames.append(pd.DataFrame({
                "strategy": strategy,
                "seed": seed,
                "iteration": df["iteration"].astype(np.int64),
                "num_labeled": df["num_labeled_samples"].astype(np.int64),
                "query_time": df["query_time"].astype(np.float64),
                "test_mAP": df["test_mAP"].astype(np.float64),
                "rare_mAP": df[rare_columns].astype(np.float64).mean(axis=1),
                "selection": [json.loads(cell) for cell in df[SELECTION_COLUMN]],
            }))
    if skipped:
        warnings.warn("skipped incomplete seeds (results.csv has fewer rows than max_iter):\n  " + "\n  ".join(skipped),
                      stacklevel=2)
    if not frames:
        raise ValueError("no complete seed to load")
    return _finalize_curves(pd.concat(frames, ignore_index=True))


def load_seg_labels(cache_dir) -> pd.DataFrame:
    """Load the segment-level targets of the training pool as a class-named boolean table.

    Args:
        cache_dir: Dataset cache directory holding meta.json and the npy arrays.

    Returns:
        ``[N, C]`` presence table whose columns follow the cache class order.
    """
    cache_dir = Path(cache_dir)
    class_names = json.loads((cache_dir / META_FILENAME).read_text(encoding="utf-8"))["class_names"]
    seg_labels = np.load(cache_dir / SEG_LABELS_FILE, mmap_mode="c")
    return pd.DataFrame(np.asarray(seg_labels) > 0, columns=class_names)


def rare_selection_counts(curves: pd.DataFrame, seg_labels: pd.DataFrame) -> pd.DataFrame:
    """Count the selected segments that carry each rare class, for every run.

    Args:
        curves: Long-format table from load_curves.
        seg_labels: Presence table from load_seg_labels.

    Returns:
        ``[strategy, seed, num_selected, *RARE_CLASSES]``, one row per run, in the row order of curves.
    """
    rare_presence = seg_labels[list(RARE_CLASSES)].to_numpy()
    rows = []
    for (strategy, seed), group in curves.groupby(["strategy", "seed"], observed=True, sort=False):
        # Rounds select disjoint segments, so the concatenated selections are the labeled set.
        selected = np.concatenate([np.asarray(batch, dtype=np.int64) for batch in group["selection"]])
        rows.append({
            "strategy": strategy,
            "seed": int(seed),
            "num_selected": int(group.iloc[-1]["num_labeled"]),
            **dict(zip(RARE_CLASSES, rare_presence[selected].sum(axis=0).astype(np.float64))),
        })
    return pd.DataFrame(rows)


def _normalized_aulc(xs: np.ndarray, ys: np.ndarray) -> float:
    """Return the trapezoidal integral of ys over xs divided by the span of xs, NaN for a degenerate span.

    Mirrors src.models.metrics.cumulative_aulc, kept local so that summaries do not import torch.
    """
    if xs.size < 2 or xs[-1] <= xs[0]:
        return float("nan")
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def full_supervised_target(curves: pd.DataFrame) -> float:
    """Return the mean final test mAP of the full-supervised reference runs as a fraction, NaN when absent."""
    rows = curves[curves["strategy"] == FULL_SUPERVISED]
    if rows.empty:
        return float("nan")
    return float(rows.groupby("seed", observed=True)["test_mAP"].last().mean())


def seed_metrics(
    curves: pd.DataFrame,
    *,
    seg_labels: pd.DataFrame | None = None,
    full_supervised_map: float | None = None,
) -> pd.DataFrame:
    """Reduce every run to the per-seed metrics of the paper table.

    N-AULC and Rare-N-AULC integrate test_mAP and rare_mAP over the labeled budget and divide by the
    budget span, and FS-Bud is the first budget whose test_mAP reaches the full-supervised reference.
    QT sums the query time of rounds 1 and later because round 0 draws the shared random cold-start set.

    Args:
        curves: Long-format table from load_curves.
        seg_labels: Presence table from load_seg_labels, supplying both the selected-set counts and the
            training-pool prevalence behind R-Enr. None leaves R-Enr as NaN.
        full_supervised_map: Target test mAP as a fraction, usually full_supervised_target(curves).
            None leaves FS-Bud as NaN.

    Returns:
        ``[strategy, seed, *SEED_METRICS]``, one row per run in the row order of curves, with the mAP
        metrics as percentages, FS-Bud in segments and QT in seconds.
    """
    rows = []
    for (strategy, seed), group in curves.groupby(["strategy", "seed"], observed=True, sort=False):
        budget = group["num_labeled"].to_numpy(dtype=np.float64)
        test_map = group["test_mAP"].to_numpy(dtype=np.float64)
        rare_map = group["rare_mAP"].to_numpy(dtype=np.float64)
        query_time = group["query_time"].to_numpy(dtype=np.float64)[group["iteration"].to_numpy() > 0]
        reached = (
            np.flatnonzero(test_map >= full_supervised_map) if full_supervised_map is not None
            else np.empty(0, dtype=np.int64)
        )
        rows.append({
            "strategy": strategy,
            "seed": int(seed),
            "N-AULC": 100.0 * _normalized_aulc(budget, test_map),
            "Rare-N-AULC": 100.0 * _normalized_aulc(budget, rare_map),
            "F-mAP": 100.0 * test_map[-1],
            "F-rmAP": 100.0 * rare_map[-1],
            "R-Enr": float("nan"),
            "FS-Bud": float(budget[reached[0]]) if reached.size else float("nan"),
            "QT": float(query_time.sum()),
        })
    metrics = pd.DataFrame(rows, columns=["strategy", "seed", *SEED_METRICS])
    metrics["strategy"] = pd.Categorical(metrics["strategy"], categories=list(CURVE_STRATEGY_ORDER), ordered=True)
    if seg_labels is not None:
        # rare_selection_counts walks the same groups in the same order, so the rows line up positionally.
        counts = rare_selection_counts(curves, seg_labels)
        selected_prevalence = (
            counts[list(RARE_CLASSES)].to_numpy(dtype=np.float64)
            / counts[["num_selected"]].to_numpy(dtype=np.float64)
        )
        pool_prevalence = seg_labels[list(RARE_CLASSES)].to_numpy().mean(axis=0)
        metrics["R-Enr"] = (selected_prevalence / pool_prevalence).mean(axis=1)
    return metrics


def summary_table(
    curves: pd.DataFrame,
    seg_labels: pd.DataFrame | None,
    full_supervised_map: float,
) -> pd.DataFrame:
    """Aggregate the per-seed metrics into the paper table, one row per strategy.

    Args:
        curves: Long-format table from load_curves.
        seg_labels: Presence table from load_seg_labels. None leaves R-Enr as NaN.
        full_supervised_map: Target test mAP as a fraction, usually full_supervised_target(curves).

    Returns:
        ``[strategy, n_seeds, {metric}_mean, {metric}_std ...]`` over SUMMARY_METRICS, in
        STRATEGY_ORDER and omitting strategies without runs. Standard deviations use ddof=1.
        FS-Bud averages the runs that reach the target and is NaN when none does. FS-Rch is the
        percentage of such runs and has no standard deviation. Both are NaN without a reference level.
    """
    metrics = seed_metrics(curves, seg_labels=seg_labels, full_supervised_map=full_supervised_map)
    reach_defined = full_supervised_map is not None and np.isfinite(full_supervised_map)
    rows = []
    for strategy in STRATEGY_ORDER:
        block = metrics[metrics["strategy"] == strategy]
        if block.empty:
            continue
        row = {"strategy": strategy, "n_seeds": int(len(block))}
        for metric in ("N-AULC", "Rare-N-AULC", "F-mAP", "F-rmAP", "R-Enr", "QT"):
            values = block[metric]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        budgets = block["FS-Bud"].dropna()
        row["FS-Bud_mean"] = float(budgets.mean()) if len(budgets) else float("nan")
        row["FS-Bud_std"] = float(budgets.std()) if len(budgets) > 1 else float("nan")
        # FS-Rch counts the runs reaching the reference level, so it is undefined without one.
        row["FS-Rch_mean"] = 100.0 * len(budgets) / len(block) if reach_defined else float("nan")
        row["FS-Rch_std"] = float("nan")
        rows.append(row)
    columns = ["strategy", "n_seeds"]
    for metric in SUMMARY_METRICS:
        columns += [f"{metric}_mean", f"{metric}_std"]
    return pd.DataFrame(rows, columns=columns)


def _paired_values(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot per-seed metric values into a ``seed x strategy`` table.

    Args:
        frame: Long-format curves, or a seed_metrics table when the metric needs the cache or the
            full-supervised target.
        metric: Metric name in SEED_METRICS.

    Returns:
        Values indexed by seed with one column per strategy, missing pairs left as NaN.

    Raises:
        ValueError: The metric is not a per-seed metric.
    """
    if metric not in SEED_METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {SEED_METRICS}")
    table = frame if metric in frame.columns else seed_metrics(frame)
    return table.astype({"strategy": str}).pivot(index="seed", columns="strategy", values=metric)


def holm_correction(p_values) -> np.ndarray:
    """Apply the Holm-Bonferroni step-down correction, keeping the input order.

    Args:
        p_values: Raw p-values of the comparisons corrected together.

    Returns:
        Adjusted p-values, monotone along ascending raw p-values and clipped to 1.
    """
    raw = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(raw, kind="stable")
    scaled = (raw.size - np.arange(raw.size)) * raw[order]
    adjusted = np.empty_like(scaled)
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def _sign_flip_p_value(diff: np.ndarray) -> float:
    """Return the two-sided p-value of a paired permutation test on the differences.

    Args:
        diff: Paired differences, one per seed.

    Returns:
        The fraction of sign patterns whose mean is at least as extreme as the observed mean.
    """
    n_pairs = diff.size
    observed = abs(float(diff.mean()))
    if n_pairs > EXACT_PERMUTATION_MAX_PAIRS:
        result = permutation_test(
            (diff,), lambda sample, axis=-1: np.mean(sample, axis=axis), permutation_type="samples",
            alternative="two-sided", n_resamples=PERMUTATION_RESAMPLES, rng=PERMUTATION_RNG_SEED,
        )
        return float(result.pvalue)
    # Row i of signs is the sign pattern encoded by the bits of i, so the enumeration is exhaustive.
    signs = 1.0 - 2.0 * ((np.arange(2 ** n_pairs)[:, None] >> np.arange(n_pairs)) & 1)
    permuted = signs @ diff / n_pairs
    return float((np.abs(permuted) >= observed - PERMUTATION_TIE_TOL * max(observed, 1.0)).mean())


def paired_permutation(
    curves: pd.DataFrame,
    metric: str = "N-AULC",
    reference: str = REFERENCE_STRATEGY,
) -> pd.DataFrame:
    """Compare the reference strategy with the others by a two-sided paired permutation test.

    The null distribution flips the sign of each paired per-seed difference. All ``2**n_pairs`` patterns
    are enumerated up to EXACT_PERMUTATION_MAX_PAIRS pairs, and random patterns are sampled beyond that.

    Args:
        curves: Long-format curves, or a seed_metrics table for metrics that need the cache or the
            full-supervised target.
        metric: Metric name in SEED_METRICS.
        reference: Reference strategy name.

    Returns:
        ``[reference, strategy, n_pairs, mean_diff, p_value, p_holm]`` in STRATEGY_ORDER, where
        mean_diff is the mean of reference minus opponent and p_holm corrects across the comparisons
        of this call.
    """
    table = _paired_values(curves, metric)
    rows = []
    for strategy in STRATEGY_ORDER:
        if strategy == reference or strategy not in table or reference not in table:
            continue
        pair = table[[reference, strategy]].dropna()
        if pair.empty:
            continue
        diff = (pair[reference] - pair[strategy]).to_numpy(dtype=np.float64)
        rows.append({
            "reference": reference,
            "strategy": strategy,
            "n_pairs": int(diff.size),
            "mean_diff": float(diff.mean()),
            "p_value": _sign_flip_p_value(diff),
        })
    result = pd.DataFrame(rows, columns=["reference", "strategy", "n_pairs", "mean_diff", "p_value"])
    result["p_holm"] = holm_correction(result["p_value"].to_numpy(dtype=np.float64))
    return result


def paired_wilcoxon(
    curves: pd.DataFrame,
    metric: str = "N-AULC",
    reference: str = REFERENCE_STRATEGY,
) -> pd.DataFrame:
    """Compare the reference strategy with the others by an exact two-sided Wilcoxon signed-rank test.

    Zero differences are excluded through scipy's default ``zero_method="wilcox"``.

    Args:
        curves: Long-format curves, or a seed_metrics table for metrics that need the cache or the
            full-supervised target.
        metric: Metric name in SEED_METRICS.
        reference: Reference strategy name.

    Returns:
        ``[reference, strategy, n_pairs, wins, mean_diff, statistic, p_value]`` in STRATEGY_ORDER, where
        wins counts the seeds the reference strictly wins, mean_diff is the mean of reference minus
        opponent, and statistic is min(W+, W-).
    """
    table = _paired_values(curves, metric)
    rows = []
    for strategy in STRATEGY_ORDER:
        if strategy == reference or strategy not in table or reference not in table:
            continue
        pair = table[[reference, strategy]].dropna()
        if pair.empty:
            continue
        diff = pair[reference] - pair[strategy]
        result = wilcoxon(pair[reference], pair[strategy], alternative="two-sided", method="exact")
        rows.append({
            "reference": reference,
            "strategy": strategy,
            "n_pairs": int(len(pair)),
            "wins": int((diff > 0).sum()),
            "mean_diff": float(diff.mean()),
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        })
    return pd.DataFrame(
        rows, columns=["reference", "strategy", "n_pairs", "wins", "mean_diff", "statistic", "p_value"]
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a Markdown table from preformatted cells.

    Args:
        headers: Column headers.
        rows: Rows of cell strings, each as long as headers.

    Returns:
        The table as a single string without a trailing newline.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
