from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    # Allow running the module directly via `python plots/...py` by adding project root to path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.detect_preconditioner_columns import detect_preconditioner_columns

PathLike = Union[str, Path]


def plot_convergence_ratio_vs_condition_matplotlib(
    df: pd.DataFrame,
    title: str = "Normalized Iteration Count vs Condition Number",
    figsize: Tuple[int, int] = (12, 8),
    font_size: int = 12,
    title_font_size: int = 14,
    legend_font_size: int = 10,
    axis_label_font_size: int = 12,
    show_legend: bool = True,
    legend_bbox: Tuple[float, float] = (1.05, 1),
    legend_loc: str = "upper left",
    output_dir: PathLike = Path("outputs"),
    output_filename: str | None = None,
) -> Path | None:
    """Plot normalized iteration count against condition number using Matplotlib."""
    col_map = detect_preconditioner_columns(df)

    if col_map["is_converged"]:
        df_converged = df[df[col_map["is_converged"]] == 1].copy()
    else:
        df_converged = df.copy()

    if len(df_converged) == 0:
        print("No data to plot")
        return None

    # Calculate normalized iteration count if required columns exist
    normalized_col = "normalized_iteration_count"
    iters_col = col_map["iters"]
    max_iters_col = col_map["max_iters"]

    if iters_col and max_iters_col:
        nonzero_mask = df_converged[max_iters_col] != 0
        if not nonzero_mask.all():
            if "matrix_name" in df_converged.columns:
                identifiers = df_converged.loc[~nonzero_mask, "matrix_name"].astype(str)
            else:
                identifiers = df_converged.index.astype(str)
            missing_preview = ", ".join(sorted(identifiers.unique()[:5]))
            print(
                "Warning: rows with zero max_iters removed: "
                f"{missing_preview}{'...' if (~nonzero_mask).sum() > 5 else ''}"
            )
            df_converged = df_converged[nonzero_mask].copy()

        df_converged[normalized_col] = (
            df_converged[iters_col] / df_converged[max_iters_col]
        )
    else:
        print("Required columns for normalized iteration count not found")
        return None

    if len(df_converged) == 0:
        print("No data to plot after normalization step")
        return None

    condition_col = col_map["condition_number"]
    if not condition_col:
        print("Condition number column not found")
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create scatter plot with colors for different configs
    config_col = col_map["config"]

    if config_col:
        configs = df_converged[config_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

        for i, config in enumerate(configs):
            config_data = df_converged[df_converged[config_col] == config]
            ax.scatter(
                config_data[condition_col],
                config_data[normalized_col],
                c=[colors[i]],
                label=config[:20] + "..." if len(config) > 20 else config,
                alpha=0.6,
                s=30,
            )
    else:
        ax.scatter(
            df_converged[condition_col],
            df_converged[normalized_col],
            alpha=0.6,
            s=30,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Condition Number", fontsize=axis_label_font_size)
    ax.set_ylabel(
        "Normalized Iteration Count (iters / max_iters)",
        fontsize=axis_label_font_size,
    )
    ax.set_title(title, fontsize=title_font_size)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    if show_legend and config_col:
        ax.legend(bbox_to_anchor=legend_bbox, loc=legend_loc, fontsize=legend_font_size)

    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_filename = output_filename or f"{title}.svg"
    output_path = output_dir / target_filename

    plt.savefig(output_path, format="svg")
    plt.close(fig)

    return output_path
