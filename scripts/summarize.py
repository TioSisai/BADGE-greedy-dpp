"""Summarize the main active learning table and paired significance tests as Markdown/CSV in OUTPUT_ROOT/tables/."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.results import (  # noqa: E402
    CURVE_STRATEGY_ORDER,
    DATASET_NAMES,
    DISPLAY_NAMES,
    FULL_SUPERVISED,
    META_FILENAME,
    REFERENCE_STRATEGY,
    STRATEGY_ORDER,
    discover_config_dirs,
    full_supervised_target,
    load_curves,
    load_seg_labels,
    markdown_table,
    paired_permutation,
    paired_wilcoxon,
    parse_config_filter,
    seed_metrics,
    summary_table,
)

# Metrics of the main table with the number of decimals they are printed with.
MAIN_METRICS = (
    ("N-AULC", 1), ("Rare-N-AULC", 1), ("F-mAP", 1), ("F-rmAP", 1),
    ("R-Enr", 2), ("FS-Bud", 0), ("FS-Rch", 0), ("QT", 2),
)
MAIN_HEADERS = ("Method", "N-AULC", "Rare-N-AULC", "F-mAP", "F-rmAP", "R-Enr", "FS-Bud", "FS-Rch", "QT (s)")
# Metrics whose best value is the largest. FS-Bud is best when smallest, and machine-dependent QT is never marked.
HIGHER_IS_BETTER = ("N-AULC", "Rare-N-AULC", "F-mAP", "F-rmAP", "R-Enr", "FS-Rch")

# Metrics of the paired tests. FS-Bud is undefined for runs below the reference and QT is machine dependent.
TEST_METRICS = ("N-AULC", "Rare-N-AULC", "F-mAP", "F-rmAP", "R-Enr")
PERMUTATION_HEADERS = ("Comparison", "Metric", "Pairs", "Mean diff.", "p (exact)", "p (Holm)")
WILCOXON_HEADERS = ("Comparison", "Metric", "Seeds won", "Mean diff.", "W", "p (two-sided, exact)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse summary arguments, defaulting cache and output paths to CACHE_ROOT and OUTPUT_ROOT."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=DATASET_NAMES[0], choices=list(DATASET_NAMES),
                        help=f"dataset to summarize (default {DATASET_NAMES[0]})")
    parser.add_argument("--reference", default=REFERENCE_STRATEGY, choices=list(STRATEGY_ORDER),
                        help=f"reference strategy of the paired tests (default {REFERENCE_STRATEGY})")
    parser.add_argument("--config-hash", nargs="+", default=None, metavar="PREFIX",
                        help="keep only config_hash dirs with these name prefixes (per-strategy hash, several allowed)")
    parser.add_argument("--config-filter", nargs="+", default=[], metavar="KEY=VALUE",
                        help="keep only config_hash dirs whose config.json fields equal these values")
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT"),
                        help="frame-level cache root (default CACHE_ROOT in .env), read for the train-pool "
                             "prevalence behind R-Enr; R-Enr is left blank when the cache is missing")
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"),
                        help="experiment output root (default OUTPUT_ROOT in .env), tables written to its tables/")
    return parser.parse_args(argv)


def load_seg_labels_or_none(cache_root, dataset: str) -> pd.DataFrame | None:
    """Read the segment-level targets of the training pool that R-Enr needs.

    Args:
        cache_root: Frame-level cache root, possibly None.
        dataset: Dataset name.

    Returns:
        The presence table, or None when the cache is missing, in which case R-Enr is left blank.
    """
    cache_dir = Path(cache_root) / dataset if cache_root else None
    if cache_dir is None or not (cache_dir / META_FILENAME).is_file():
        print(f"[note] frame-level cache not found under {cache_dir}, R-Enr left blank", file=sys.stderr)
        return None
    return load_seg_labels(cache_dir)


def load_run_curves(args: argparse.Namespace) -> pd.DataFrame | None:
    """Load the per-round results of a reproduced run, returning None when no strategy has outputs yet."""
    config_dirs = discover_config_dirs(
        Path(args.output_root), args.dataset, CURVE_STRATEGY_ORDER,
        config_hashes=args.config_hash, config_filter=parse_config_filter(args.config_filter),
    )
    if not config_dirs:
        return None
    return load_curves(config_dirs)


def _format_cell(mean: float, std: float, decimals: int, *, percent: bool = False) -> str:
    """Render a ``mean ± std`` cell, printing -- for an undefined mean and dropping an undefined deviation."""
    if not np.isfinite(mean):
        return "--"
    if percent:
        return f"{mean:.{decimals}f}%"
    if not np.isfinite(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _best_values(summary: pd.DataFrame) -> dict[str, float]:
    """Return the best mean of every marked metric, leaving out metrics whose column is entirely undefined."""
    best = {}
    for metric in (*HIGHER_IS_BETTER, "FS-Bud"):
        values = summary[f"{metric}_mean"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).any():
            continue
        best[metric] = float(np.nanmin(values) if metric == "FS-Bud" else np.nanmax(values))
    return best


def build_main_table(summary: pd.DataFrame) -> str:
    """Format the summary as the main table of the paper, bolding the best value of every marked metric.

    Args:
        summary: Return value of summary_table.

    Returns:
        The table as Markdown, one row per strategy in STRATEGY_ORDER.
    """
    best = _best_values(summary)
    rows = []
    for record in summary.to_dict("records"):
        row = [DISPLAY_NAMES[record["strategy"]]]
        for metric, decimals in MAIN_METRICS:
            mean, std = record[f"{metric}_mean"], record[f"{metric}_std"]
            text = _format_cell(mean, std, decimals, percent=metric == "FS-Rch")
            row.append(f"**{text}**" if mean == best.get(metric) else text)
        rows.append(row)
    return markdown_table(MAIN_HEADERS, rows)


def build_permutation_table(metrics: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, str]:
    """Run the Holm-corrected paired permutation tests of every tested metric.

    Args:
        metrics: Per-seed metrics from seed_metrics.
        reference: Reference strategy name.

    Returns:
        ``(table, markdown)``, the concatenated test results and their Markdown rendering, grouped by
        metric with every group in STRATEGY_ORDER.
    """
    blocks = []
    for metric in TEST_METRICS:
        result = paired_permutation(metrics, metric, reference)
        if result.empty:
            continue
        result.insert(0, "metric", metric)
        blocks.append(result)
    if not blocks:
        return pd.DataFrame(), ""
    table = pd.concat(blocks, ignore_index=True)
    rows = [
        [
            f"{DISPLAY_NAMES[row.reference]} vs {DISPLAY_NAMES[row.strategy]}",
            row.metric,
            str(row.n_pairs),
            f"{row.mean_diff:+.3f}",
            f"{row.p_value:.4g}",
            f"{row.p_holm:.4g}",
        ]
        for row in table.itertuples(index=False)
    ]
    return table, markdown_table(PERMUTATION_HEADERS, rows)


def build_wilcoxon_table(metrics: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, str]:
    """Run the exact paired Wilcoxon signed-rank tests of every tested metric.

    Args:
        metrics: Per-seed metrics from seed_metrics.
        reference: Reference strategy name.

    Returns:
        ``(table, markdown)``, the concatenated test results and their Markdown rendering, grouped by
        metric with every group in STRATEGY_ORDER.
    """
    blocks = []
    for metric in TEST_METRICS:
        result = paired_wilcoxon(metrics, metric, reference)
        if result.empty:
            continue
        result.insert(0, "metric", metric)
        blocks.append(result)
    if not blocks:
        return pd.DataFrame(), ""
    table = pd.concat(blocks, ignore_index=True)
    rows = [
        [
            f"{DISPLAY_NAMES[row.reference]} vs {DISPLAY_NAMES[row.strategy]}",
            row.metric,
            f"{row.wins}/{row.n_pairs}",
            f"{row.mean_diff:+.3f}",
            f"{row.statistic:g}",
            f"{row.p_value:.4g}",
        ]
        for row in table.itertuples(index=False)
    ]
    return table, markdown_table(WILCOXON_HEADERS, rows)


def build_sections(args: argparse.Namespace) -> dict[str, tuple[pd.DataFrame, str]] | None:
    """Build the three tables of one reproduced run.

    Args:
        args: Parsed command line.

    Returns:
        ``{kind: (table, markdown_section)}`` over main, permutation and wilcoxon, omitting the tests when
        the reference strategy has no runs, and None when there are no results at all.
    """
    curves = load_run_curves(args)
    if curves is None:
        return None
    seg_labels = load_seg_labels_or_none(args.cache_root, args.dataset)

    present = set(curves["strategy"].astype(str))
    if missing := [s for s in STRATEGY_ORDER if s not in present]:
        print(f"[note] missing strategies {', '.join(missing)}", file=sys.stderr)
    target = full_supervised_target(curves)
    if not np.isfinite(target):
        print(f"[note] no {FULL_SUPERVISED} run, FS-Bud and FS-Rch are left empty", file=sys.stderr)
    summary = summary_table(curves, seg_labels, target)
    if summary.empty:
        return None
    metrics = seed_metrics(curves, seg_labels=seg_labels, full_supervised_map=target)

    main_note = (
        f"\n\nFS-Bud and FS-Rch are measured against the mean final test mAP of the {FULL_SUPERVISED} "
        f"reference, {100.0 * target:.2f}%." if np.isfinite(target) else ""
    )
    sections = {"main": (summary, f"## {args.dataset} main table (mean ± std over seeds)\n\n"
                                  f"{build_main_table(summary)}{main_note}\n")}
    if args.reference not in present:
        print(f"[note] reference strategy {args.reference} has no runs, paired tests skipped", file=sys.stderr)
        return sections
    reference_name = DISPLAY_NAMES[args.reference]
    for kind, heading, builder in (
        ("permutation", f"Holm-corrected paired permutation tests of {reference_name}", build_permutation_table),
        ("wilcoxon", f"exact paired Wilcoxon signed-rank tests of {reference_name}", build_wilcoxon_table),
    ):
        table, markdown = builder(metrics, args.reference)
        if not table.empty:
            sections[kind] = (table, f"## {args.dataset} {heading}\n\n{markdown}\n")
    return sections


def _atomic_write_text(path: Path, text: str) -> None:
    """Write to a temporary file in the same directory, then atomically replace the target."""
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.output_root:
        raise SystemExit("missing --output-root (and .env provides no OUTPUT_ROOT)")
    output_root = Path(args.output_root)

    sections = build_sections(args)
    if sections is None:
        raise SystemExit(f"no experiment outputs under {output_root / args.dataset}")

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for kind, (table, markdown) in sections.items():
        _atomic_write_text(table_dir / f"{args.dataset}_{kind}.md", markdown)
        _atomic_write_text(table_dir / f"{args.dataset}_{kind}.csv", table.to_csv(index=False))
        print(markdown)
    print(f"Tables written to {table_dir}")


if __name__ == "__main__":
    main()
