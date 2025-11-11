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

    counts = working.groupby([config_col, status_col]).size().unstack(fill_value=0)
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
    status_order = preferred_order + [
        status for status in frequency_order if status not in seen
    ]

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
        sort_keys["_total"] = totals
        config_order = sort_keys.sort_values(
            by=[sort_target, "_total"], ascending=[False, False]
        ).index.tolist()
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

        if min_fraction_label is not None:
            centers = left + values / 2
            for bar_idx, (value, center) in enumerate(zip(values, centers)):
                count_value = count_values[bar_idx]
                if count_value < min_count_label or value <= 0:
                    continue

                # Skip labels when the slice is too narrow.
                visible_threshold = max(min_fraction_label or 0.0, 0.12)
                if value < visible_threshold:
                    continue

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

    savefig_kwargs: dict[str, str] = {}
    file_format = output_path.suffix.lstrip(".").lower()
    if file_format:
        savefig_kwargs["format"] = file_format

    plt.savefig(output_path, **savefig_kwargs)
    plt.close(fig)

    return output_path


def categorize_by_condition_number(
    df: pd.DataFrame,
    cond_col: str = "condition_number",
) -> dict[str, pd.DataFrame]:
    """Categorize dataframe by condition number ranges.

    Returns:
        Dictionary with keys 'low_cond', 'mid_cond', 'high_cond' containing filtered dataframes.
    """
    if cond_col not in df.columns:
        raise ValueError(f"Column '{cond_col}' not found in dataframe")

    # Filter out rows with missing condition numbers
    df_valid = df[~df[cond_col].isna()].copy()

    categories = {
        "low_cond": df_valid[(df_valid[cond_col] >= 1.0) & (df_valid[cond_col] < 1e5)],
        "mid_cond": df_valid[(df_valid[cond_col] >= 1e5) & (df_valid[cond_col] < 1e10)],
        "high_cond": df_valid[df_valid[cond_col] >= 1e10],
    }

    return categories


def plot_by_condition_number_ranges(
    df: pd.DataFrame,
    base_title: str = "Solve Status Ratio by Preconditioner",
    status_priority: Sequence[str] | None = None,
    output_dir: PathLike = Path("outputs"),
    **plot_kwargs,
) -> dict[str, Path | None]:
    """Create plots for each condition number range.

    Returns:
        Dictionary mapping category names to output paths.
    """
    categories = categorize_by_condition_number(df)

    category_labels = {
        "low_cond": r"Low Condition ($1 \leq \kappa_2(A) < 10^5$)",
        "mid_cond": r"Mid Condition ($10^5 \leq \kappa_2(A) < 10^{10}$)",
        "high_cond": r"High Condition ($10^{10} \leq \kappa_2(A)$)",
    }

    output_paths = {}

    for category, df_subset in categories.items():
        if df_subset.empty:
            print(f"No data for category '{category}', skipping plot")
            output_paths[category] = None
            continue

        label = category_labels[category]
        title = f"{base_title} - {label}"
        filename = f"{base_title.replace(' ', '_').lower()}_{category}.svg"

        print(f"Generating plot for {category}: {len(df_subset)} rows")

        output_path = plot_solve_status_ratio_by_prec_label(
            df_subset,
            title=title,
            status_priority=status_priority,
            output_dir=output_dir,
            output_filename=filename,
            **plot_kwargs,
        )

        output_paths[category] = output_path

        if output_path:
            print(f"  → Saved to {output_path}")

    return output_paths


if __name__ == "__main__":
    runs_path = Path("inputs/runs.csv")
    dataframe = pd.read_csv(runs_path)

    print("Generating plots by condition number ranges...")
    output_paths = plot_by_condition_number_ranges(
        dataframe,
        base_title="Solve Status Ratio by Preconditioner",
        status_priority=[
            "reached_tol",
            "max_iterations",
            "prec_factorization_breakdown",
            "prec_unknown",
        ],
    )

    print("\nSummary:")
    for category, path in output_paths.items():
        if path:
            print(f"  {category}: {path}")
        else:
            print(f"  {category}: No plot generated")
