"""Export the rare call type enrichment bars of Figure 2 to OUTPUT_ROOT/figures/."""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.results import (  # noqa: E402
    DATASET_NAMES,
    DISPLAY_NAMES,
    META_FILENAME,
    RARE_CLASSES,
    STRATEGY_ORDER,
    discover_config_dirs,
    load_curves,
    load_seg_labels,
    parse_config_filter,
    rare_selection_counts,
)
from src.analysis.style import apply_nature_style, mm, save_figure  # noqa: E402

# One color per rare call type, distinct from the strategy colors.
RARE_CLASS_COLORS = {"gwl": "#4C72B0", "snr": "#DD8452", "str": "#55A868"}
BAR_WIDTH = 0.26
# Wrap width and font size that keep the strategy tick labels inside one bar group.
LABEL_WRAP = 16
LABEL_FONT_SIZE = 5.5
# Selecting segments at the prevalence of the training pool gives an enrichment of exactly one.
NEUTRAL_ENRICHMENT = 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse plotting arguments, defaulting cache and output paths to CACHE_ROOT and OUTPUT_ROOT."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=DATASET_NAMES[0], choices=list(DATASET_NAMES),
                        help=f"dataset to plot (default {DATASET_NAMES[0]})")
    parser.add_argument("--config-hash", nargs="+", default=None, metavar="PREFIX",
                        help="keep only config_hash dirs with these name prefixes (per-strategy hash, several allowed)")
    parser.add_argument("--config-filter", nargs="+", default=[], metavar="KEY=VALUE",
                        help="keep only config_hash dirs whose config.json fields equal these values")
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "png"], help="export formats")
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT"),
                        help="frame-level cache root (default CACHE_ROOT in .env), read for the train-pool "
                             "prevalence the enrichment divides by, and for the segment labels that count the "
                             "selections of a reproduced run")
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"),
                        help="experiment output root (default OUTPUT_ROOT in .env), figures written to its figures/")
    return parser.parse_args(argv)


def load_pool_labels(cache_root, dataset: str) -> pd.DataFrame:
    """Read the segment-level targets of the training pool that the enrichment is measured against.

    Args:
        cache_root: Frame-level cache root, possibly None.
        dataset: Dataset name.

    Returns:
        The presence table from load_seg_labels.

    Raises:
        SystemExit: The cache is missing, so the training-pool prevalence is unavailable.
    """
    if not cache_root:
        raise SystemExit("missing --cache-root (and .env provides no CACHE_ROOT)")
    cache_dir = Path(cache_root) / dataset
    if not (cache_dir / META_FILENAME).is_file():
        raise SystemExit(f"frame-level cache not found: {cache_dir} (download it from Zenodo, see README)")
    return load_seg_labels(cache_dir)


def load_run_curves(args: argparse.Namespace) -> pd.DataFrame | None:
    """Load the per-round results of the query strategies, returning None when there are no outputs yet.

    The full-supervised reference has no selection to enrich and is left out.
    """
    config_dirs = discover_config_dirs(
        Path(args.output_root), args.dataset, STRATEGY_ORDER,
        config_hashes=args.config_hash, config_filter=parse_config_filter(args.config_filter),
    )
    return load_curves(config_dirs) if config_dirs else None


def enrichment_per_run(curves: pd.DataFrame, seg_labels: pd.DataFrame) -> pd.DataFrame:
    """Divide the rare-class prevalence of every selected set by its prevalence in the training pool.

    Args:
        curves: Long-format table restricted to the query strategies.
        seg_labels: Presence table from load_seg_labels.

    Returns:
        ``[*RARE_CLASSES, strategy]``, one row per run.
    """
    counts = rare_selection_counts(curves, seg_labels)
    selected_prevalence = (
        counts[list(RARE_CLASSES)].to_numpy(dtype=np.float64)
        / counts[["num_selected"]].to_numpy(dtype=np.float64)
    )
    enrichment = selected_prevalence / seg_labels[list(RARE_CLASSES)].to_numpy().mean(axis=0)
    return pd.DataFrame(enrichment, columns=list(RARE_CLASSES)).assign(
        strategy=counts["strategy"].astype(str).to_numpy()
    )


def _wrap_label(name: str) -> str:
    """Wrap a display name so the tick labels stay upright and narrower than a bar group."""
    return textwrap.fill(name, LABEL_WRAP)


def make_figure(enrichment: pd.DataFrame):
    """Lay out Figure 2 as grouped bars over the strategies and return the matplotlib Figure.

    Args:
        enrichment: Per-run enrichment from enrichment_per_run.

    Returns:
        matplotlib Figure.

    Raises:
        SystemExit: No query strategy has results.
    """
    present = set(enrichment["strategy"])
    strategies = [s for s in STRATEGY_ORDER if s in present]
    if not strategies:
        raise SystemExit("no query strategy has results to plot")
    grouped = enrichment.groupby("strategy", sort=False)[list(RARE_CLASSES)]
    means = grouped.mean().reindex(strategies)
    # Sample standard deviation over seeds, matching the tables.
    stds = grouped.std().reindex(strategies)

    fig, ax = plt.subplots(figsize=(mm(183), mm(76)))
    positions = np.arange(len(strategies), dtype=np.float64)
    for index, name in enumerate(RARE_CLASSES):
        offset = (index - 0.5 * (len(RARE_CLASSES) - 1)) * BAR_WIDTH
        ax.bar(positions + offset, means[name].to_numpy(), BAR_WIDTH,
               yerr=stds[name].to_numpy(), color=RARE_CLASS_COLORS[name], linewidth=0,
               label=name.upper(), zorder=3,
               error_kw={"elinewidth": 0.7, "capsize": 1.6, "capthick": 0.7, "ecolor": "#333333", "zorder": 4})
    ax.axhline(NEUTRAL_ENRICHMENT, color="#444444", linestyle="--", linewidth=0.9, zorder=2)
    ax.annotate("prevalence-matched\nselection", xy=(1.0, NEUTRAL_ENRICHMENT),
                xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=plt.rcParams["legend.fontsize"], color="#444444")

    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(DISPLAY_NAMES[s]) for s in strategies], fontsize=LABEL_FONT_SIZE)
    ax.set_xlim(positions[0] - 0.55, positions[-1] + 0.55)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Segment-level (selection) enrichment")
    ax.grid(True, which="major", axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(title="Rare call type", loc="upper left", handlelength=1.2, handletextpad=0.5,
              borderaxespad=0.5, labelspacing=0.35)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.output_root:
        raise SystemExit("missing --output-root (and .env provides no OUTPUT_ROOT)")
    curves = load_run_curves(args)
    if curves is None:
        raise SystemExit(f"no experiment outputs under {Path(args.output_root) / args.dataset}")
    seg_labels = load_pool_labels(args.cache_root, args.dataset)

    apply_nature_style(font_size=7)
    fig = make_figure(enrichment_per_run(curves, seg_labels))
    saved = save_figure(fig, f"fig2_enrichment_{args.dataset}", Path(args.output_root) / "figures", args.formats)
    plt.close(fig)
    print("Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
