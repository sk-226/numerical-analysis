from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    # Allow running the module directly via `python plots/...py` by adding project root to path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.detect_preconditioner_columns import detect_preconditioner_columns

PathLike = Union[str, Path]


def compute_solve_status_ratios(
    df: pd.DataFrame,
    config_col: str | None = None,
    status_col: str = "solve_status",
    status_priority: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return solve-status counts and row-wise ratios grouped by preconditioner."""
    if status_col not in df.columns:
        raise ValueError(f"Column '{status_col}' not found in dataframe")

    if config_col is None:
        config_col = detect_preconditioner_columns(df).get("config")
        if not config_col:
            raise ValueError("No preconditioner label column detected")

    working = df.copy()
    working = working[~working[config_col].isna()].copy()
    if working.empty:
        raise ValueError("No rows with preconditioner labels to aggregate")

    working[config_col] = working[config_col].astype(str)
    working[status_col] = working[status_col].fillna("unknown").astype(str)

    counts = (
        working.groupby([config_col, status_col]).size().unstack(fill_value=0)
    )
    if counts.empty:
        raise ValueError("No solve status information found after grouping")

    totals = counts.sum(axis=1)
    ratios = counts.div(totals, axis=0)

    if status_priority is None:
        status_priority = [
            "reached_tol",
            "max_iterations",
            "prec_factorization_breakdown",
            "prec_unknown",
        ]

    preferred_order: list[str] = []
    seen: set[str] = set()
    for label in status_priority:
        if label in counts.columns and label not in seen:
            preferred_order.append(label)
            seen.add(label)

    frequency_order = counts.sum(axis=0).sort_values(ascending=False).index.tolist()
    status_order = preferred_order + [status for status in frequency_order if status not in seen]

    sort_target = None
    if status_priority:
        for label in status_priority:
            if label in ratios.columns:
                sort_target = label
                break
    if sort_target is None and "reached_tol" in ratios.columns:
        sort_target = "reached_tol"

    if sort_target:
        sort_keys = ratios[[sort_target]].copy()
        sort_keys['_total'] = totals
        config_order = sort_keys.sort_values(by=[sort_target, '_total'], ascending=[False, False]).index.tolist()
    else:
        config_order = totals.sort_values(ascending=False).index.tolist()

    counts = counts.loc[config_order, status_order]
    ratios = ratios.loc[config_order, status_order]

    return counts, ratios, status_order


def plot_solve_status_ratio_by_prec_label(
    df: pd.DataFrame,
    title: str = "Solve Status Ratio by Preconditioner",
    figsize: Tuple[int, int] = (12, 6),
    font_size: int = 11,
    title_font_size: int = 14,
    legend_font_size: int = 10,
    axis_label_font_size: int = 12,
    min_fraction_label: float = 0.08,
    min_count_label: int = 1,
    status_priority: Sequence[str] | None = None,
    status_colors: Mapping[str, str] | None = None,
    output_dir: PathLike = Path("outputs"),
    output_filename: str | None = None,
) -> Path | None:
    """Create a horizontal stacked bar chart showing status ratios per preconditioner."""
    try:
        counts, ratios, status_order = compute_solve_status_ratios(
            df, status_priority=status_priority
        )
    except ValueError as err:
        print(f"Unable to produce plot: {err}")
        return None

    statuses = [status for status in status_order if status in ratios.columns]
    configs = ratios.index.tolist()

    ratios = ratios[statuses]
    counts = counts[statuses]

    if not statuses or not configs:
        print("No data to plot")
        return None

    bar_height = max(figsize[1], 0.45 * len(configs) + 1.0)
    fig, ax = plt.subplots(figsize=(figsize[0], bar_height))

    base_colors = {
        "reached_tol": "#1f77b4",
        "max_iterations": "#6baed6",
        "prec_factorization_breakdown": "#d62728",
        "prec_unknown": "#f7b6d2",
    }
    if status_colors:
        base_colors.update(status_colors)

    fallback_palette = plt.cm.tab20(np.linspace(0, 1, max(3, len(statuses))))
    fallback_idx = 0
    bar_colors: list[tuple[float, float, float, float] | str] = []
    for status in statuses:
        color = base_colors.get(status)
        if color is None:
            color = fallback_palette[min(fallback_idx, len(fallback_palette) - 1)]
            fallback_idx += 1
        bar_colors.append(color)

    left = np.zeros(len(configs), dtype=float)
    for idx, status in enumerate(statuses):
        values = ratios[status].to_numpy()
        count_values = counts[status].to_numpy()
        ax.barh(configs, values, left=left, color=bar_colors[idx], label=status)

        if min_fraction_label > 0:
            centers = left + values / 2
            for bar_idx, (value, center) in enumerate(zip(values, centers)):
                count_value = count_values[bar_idx]
                if value >= min_fraction_label and count_value >= min_count_label:
                    label = f"{int(round(count_value))}"
                    ax.text(
                        center,
                        bar_idx,
                        label,
                        va="center",
                        ha="center",
                        fontsize=font_size,
                        color="white",
                    )

        left += values

    ax.set_xlim(0, 1)
    ax.set_xlabel("Solve Status Share", fontsize=axis_label_font_size)
    ax.set_ylabel("Preconditioner", fontsize=axis_label_font_size)
    ax.set_title(title, fontsize=title_font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="x", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=legend_font_size)

    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_filename = output_filename or f"{title}.svg"
    output_path = output_dir / target_filename
    plt.savefig(output_path, format="svg")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    runs_path = Path("inputs/runs.csv")
    dataframe = pd.read_csv(runs_path)
    output_path = plot_solve_status_ratio_by_prec_label(
        dataframe,
        status_priority=[
            "reached_tol",
            "max_iterations",
            "prec_factorization_breakdown",
            "prec_unknown",
        ],
    )
    if output_path:
        print(f"Plot saved to {output_path}")
