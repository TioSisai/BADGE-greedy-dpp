"""matplotlib styling and export for paper figures."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl


def apply_nature_style(font_size: int = 7, axes_linewidth: float = 0.8) -> None:
    """Set the global style for subsequent figures.

    Args:
        font_size: Base font size (pt), increased by 1 for titles and decreased by 1 for ticks and legends.
        axes_linewidth: Line width for axes and major ticks.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",  # Preserve editable text in SVG
        "pdf.fonttype": 42,      # Embed TrueType fonts in PDF
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 600,
    })


def save_figure(
    fig,
    stem: str,
    out_dir,
    formats: Iterable[str] = ("svg", "pdf", "png"),
    dpi: int = 600,
) -> list[Path]:
    """Export each format to out_dir/{stem}.{fmt} and return the corresponding paths.

    Args:
        fig: matplotlib Figure.
        stem: Filename without an extension.
        out_dir: Output directory, created if missing.
        formats: Sequence of export formats.
        dpi: Bitmap resolution.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    return saved


def mm(length: float) -> float:
    """Convert millimeters to inches for matplotlib figsize."""
    return length / 25.4
