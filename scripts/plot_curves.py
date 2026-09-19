"""Export the active learning curves of Figure 1 to OUTPUT_ROOT/figures/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.analysis.results import (  # noqa: E402
    CURVE_STRATEGY_ORDER,
    DATASET_NAMES,
    DISPLAY_NAMES,
    FULL_SUPERVISED,
    FULL_SUPERVISED_COLOR,
    STRATEGY_COLORS,
    STRATEGY_MARKERS,
    STRATEGY_ORDER,
    discover_config_dirs,
    load_curves,
    parse_config_filter,
)
from src.analysis.style import apply_nature_style, mm, save_figure  # noqa: E402

BAND_ALPHA = 0.12
FULL_SUPERVISED_BAND_ALPHA = 0.18


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse plotting arguments, defaulting the output path to OUTPUT_ROOT."""
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
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"),
                        help="experiment output root (default OUTPUT_ROOT in .env), figures written to its figures/")
    return parser.parse_args(argv)


def load_run_curves(args: argparse.Namespace) -> pd.DataFrame | None:
    """Load the per-round results of a reproduced run, returning None when no strategy has outputs yet."""
    config_dirs = discover_config_dirs(
        Path(args.output_root), args.dataset, CURVE_STRATEGY_ORDER,
        config_hashes=args.config_hash, config_filter=parse_config_filter(args.config_filter),
    )
    return load_curves(config_dirs) if config_dirs else None


def mean_band(curves: pd.DataFrame, strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce the seeds of one strategy to a mean curve and a minimum-to-maximum band.

    Args:
        curves: Long-format table from load_curves.
        strategy: Strategy name.

    Returns:
        ``(budget, mean, low, high)`` of test_mAP over the labeled budget, in ascending budget order.
    """
    stats = (
        curves[curves["strategy"] == strategy]
        .groupby("num_labeled", observed=True)["test_mAP"]
        .agg(["mean", "min", "max"])
        .sort_index()
    )
    return (
        stats.index.to_numpy(dtype=np.float64),
        stats["mean"].to_numpy(dtype=np.float64),
        stats["min"].to_numpy(dtype=np.float64),
        stats["max"].to_numpy(dtype=np.float64),
    )


def plot_learning_curves(ax, curves: pd.DataFrame) -> list:
    """Draw one mean curve with a minimum-to-maximum band per strategy, plus the full-supervised reference.

    Args:
        ax: Target axes.
        curves: Long-format table from load_curves.

    Returns:
        Legend handles, the full-supervised reference first and the strategies in STRATEGY_ORDER.

    Raises:
        SystemExit: No query strategy has results.
    """
    strategies = [s for s in STRATEGY_ORDER if (curves["strategy"] == s).any()]
    if not strategies:
        raise SystemExit("no query strategy has results to plot")
    bands = {s: mean_band(curves, s) for s in strategies}
    budget = bands[strategies[0]][0]

    handles = []
    if (curves["strategy"] == FULL_SUPERVISED).any():
        # The reference labels the whole pool in a single round, so its band spans the whole budget axis.
        _, mean, low, high = mean_band(curves, FULL_SUPERVISED)
        ax.fill_between(budget, low[0], high[0], color=FULL_SUPERVISED_COLOR,
                        alpha=FULL_SUPERVISED_BAND_ALPHA, linewidth=0, zorder=1)
        ax.axhline(mean[0], color="black", linestyle="--", linewidth=1.1, zorder=4)
        handles.append(Line2D([], [], color="black", linestyle="--", linewidth=1.1,
                              label=DISPLAY_NAMES[FULL_SUPERVISED]))
    for strategy in strategies:
        _, _, low, high = bands[strategy]
        ax.fill_between(budget, low, high, color=STRATEGY_COLORS[strategy], alpha=BAND_ALPHA,
                        linewidth=0, zorder=2)
    for strategy in strategies:
        _, mean, _, _ = bands[strategy]
        handles.append(ax.plot(
            budget, mean, color=STRATEGY_COLORS[strategy], lw=1.3, zorder=3,
            marker=STRATEGY_MARKERS[strategy], markersize=3.8, markeredgecolor="white",
            markeredgewidth=0.4, label=DISPLAY_NAMES[strategy],
        )[0])

    margin = 0.03 * (budget[-1] - budget[0])
    ax.set_xlim(budget[0] - margin, budget[-1] + margin)
    ax.set_xticks(budget)
    ax.set_xlabel("Labeled segments")
    ax.set_ylabel("Test macro mAP")
    ax.grid(True, which="major", axis="both", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    return handles


def make_figure(curves: pd.DataFrame):
    """Lay out Figure 1 at one and a half column width and return the matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(mm(120), mm(84)))
    handles = plot_learning_curves(ax, curves)
    ax.legend(handles=handles, loc="lower right", handletextpad=0.5, borderaxespad=0.5, labelspacing=0.35)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.output_root:
        raise SystemExit("missing --output-root (and .env provides no OUTPUT_ROOT)")
    curves = load_run_curves(args)
    if curves is None:
        raise SystemExit(f"no experiment outputs under {Path(args.output_root) / args.dataset}")

    apply_nature_style(font_size=7)
    fig = make_figure(curves)
    saved = save_figure(fig, f"fig1_curves_{args.dataset}", Path(args.output_root) / "figures", args.formats)
    plt.close(fig)
    print("Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
