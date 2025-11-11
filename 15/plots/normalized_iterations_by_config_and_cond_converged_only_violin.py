from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    # Allow running the module directly via `python plots/...py` by adding project root to path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plots.convergence_rate_by_config_and_cond import categorize_by_condition_number
from utils.detect_preconditioner_columns import detect_preconditioner_columns

PathLike = Union[str, Path]


def _ensure_required_columns(col_map: Mapping[str, str | None], required: Iterable[str]) -> None:
    missing = [key for key in required if not col_map.get(key)]
    if missing:
        raise ValueError(
            "Missing required columns in dataframe: "
            + ", ".join(sorted(missing))
        )


def add_normalized_iteration_column(
    df: pd.DataFrame,
    iters_col: str,
    max_iters_col: str,
    normalized_col: str = "normalized_iteration_fraction",
) -> pd.DataFrame:
    """Return a copy of ``df`` with normalized iteration counts."""
    working = df.copy()

    working = working.dropna(subset=[iters_col, max_iters_col])

    finite_mask = np.isfinite(working[[iters_col, max_iters_col]]).all(axis=1)
    if not finite_mask.all():
        removed = working.loc[~finite_mask]
        preview = ", ".join(sorted(removed.index.astype(str)[:5]))
        print(
            "Warning: removed rows with non-finite iteration data: "
            f"{preview}{'...' if (~finite_mask).sum() > 5 else ''}"
        )
        working = working.loc[finite_mask]

    working = working.copy()

    zero_mask = working[max_iters_col] <= 0
    if zero_mask.any():
        affected = working.loc[zero_mask]
        preview = ", ".join(sorted(affected.index.astype(str)[:5]))
        print(
            "Warning: removed rows with non-positive max iterations: "
            f"{preview}{'...' if zero_mask.sum() > 5 else ''}"
        )
        working = working.loc[~zero_mask].copy()

    working[normalized_col] = working[iters_col] / working[max_iters_col]
    working = working.replace([np.inf, -np.inf], np.nan)
    working = working.dropna(subset=[normalized_col])

    return working


def _format_config_labels(configs: Iterable[str], counts: Dict[str, int]) -> list[str]:
    return [f"{config}\n(n={counts[config]})" for config in configs]


def _plot_category_violin(
    df: pd.DataFrame,
    config_col: str,
    normalized_col: str,
    category_key: str,
    category_label: str,
    base_title: str,
    output_dir: PathLike,
    output_name_template: str,
    figsize_per_config: float = 0.45,
    *,
    x_scale: str = "linear",
    output_suffix: str = "",
) -> Path | None:
    grouped: Dict[str, np.ndarray] = {}
    for config, series in df.groupby(config_col)[normalized_col]:
        values = series.dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        grouped[config] = values

    if not grouped:
        print(f"No data available for category '{category_key}' after grouping; skipping plot")
        return None

    # Order configurations by median normalized iteration, then by name for stability.
    def _median(values: np.ndarray) -> float:
        return float(np.median(values)) if len(values) else float("inf")

    config_order = sorted(grouped.keys(), key=lambda item: (_median(grouped[item]), item))

    data = [grouped[config] for config in config_order]
    counts = {config: len(grouped[config]) for config in config_order}

    # Determine figure height dynamically for readability.
    fig_height = max(4.5, figsize_per_config * len(config_order) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    tick_labels = _format_config_labels(config_order, counts)

    positions = np.arange(1, len(data) + 1, dtype=float)
    violins = ax.violinplot(
        data,
        positions=positions,
        vert=False,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        widths=0.8,
    )

    palette = plt.cm.tab20(np.linspace(0, 1, max(3, len(config_order))))
    for body, color in zip(violins["bodies"], palette[: len(violins["bodies"])]):
        body.set_facecolor(color)
        body.set_alpha(0.7)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)

    if "cmedians" in violins:
        violins["cmedians"].set_color("black")
        violins["cmedians"].set_linewidth(1.0)

    xlabel = "Normalized Iterations (iters / max_iters)"
    if x_scale == "log":
        xlabel += " [log scale]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Preconditioner")
    ax.set_yticks(positions)
    ax.set_yticklabels(tick_labels)
    title = f"{base_title} - {category_label}"
    ax.set_title(title)

    max_value = max(float(values.max()) for values in data)
    x_max = max(1.05, max_value * 1.1)

    grid_kwargs = {"axis": "x", "alpha": 0.3}
    if x_scale == "log":
        positive_mins: list[float] = []
        for values in data:
            positives = values[values > 0]
            if len(positives) == 0:
                continue
            positive_mins.append(float(positives.min()))

        if not positive_mins:
            plt.close(fig)
            print(
                f"Skipping log-scale plot for category '{category_key}' due to lack of positive values"
            )
            return None

        min_value = min(positive_mins)
        x_min = min_value * 0.8
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.grid(which="both", **grid_kwargs)
    else:
        ax.set_xlim(0, x_max)
        ax.grid(**grid_kwargs)

    # Highlight theoretical CG completion at n iterations (0.5 under max_iters = 2n)
    ax.axvline(
        0.5,
        linestyle=":",
        color="tab:red",
        alpha=0.85,
        linewidth=1.6,
        label="n iterations (0.5)",
    )

    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_name = output_name_template.format(category=category_key)
    output_path = output_dir / rendered_name
    if output_suffix:
        output_path = output_path.with_stem(output_path.stem + output_suffix)

    savefig_kwargs: Dict[str, str] = {}
    file_format = output_path.suffix.lstrip(".").lower()
    if file_format:
        savefig_kwargs["format"] = file_format

    plt.savefig(output_path, **savefig_kwargs)
    plt.close(fig)

    print(
        f"Saved violin plot for category '{category_key}' "
        f"with {len(config_order)} configurations to {output_path}"
    )

    return output_path


def plot_normalized_iterations_by_condition_category(
    df: pd.DataFrame,
    base_title: str = "Normalized Iterations by Preconditioner",
    normalized_col: str = "normalized_iteration_fraction",
    output_dir: PathLike = Path("outputs"),
    output_name_template: str = "normalized_iterations_by_prec_{category}_converged_only_violin.svg",
    scales: Iterable[str] = ("linear", "log"),
) -> Dict[str, Dict[str, Path | None]]:
    """Create violin plots of normalized iterations per preconditioner across condition categories."""
    col_map = detect_preconditioner_columns(df)
    _ensure_required_columns(
        col_map,
        required=["config", "iters", "max_iters", "condition_number"],
    )

    config_col = col_map["config"]
    iters_col = col_map["iters"]
    max_iters_col = col_map["max_iters"]
    cond_col = col_map["condition_number"]

    working = df.copy()
    working = working[~working[config_col].isna()].copy()
    working = add_normalized_iteration_column(
        working,
        iters_col=iters_col,
        max_iters_col=max_iters_col,
        normalized_col=normalized_col,
    )

    categories = categorize_by_condition_number(working, cond_col=cond_col)

    category_labels = {
        "low_cond": r"Low Condition ($1 \leq \kappa_2(A) < 10^5$)",
        "mid_cond": r"Mid Condition ($10^5 \leq \kappa_2(A) < 10^{10}$)",
        "high_cond": r"High Condition ($10^{10} \leq \kappa_2(A)$)",
    }

    scales = tuple(scales)
    valid_scales = {"linear", "log"}
    invalid_scales = sorted({scale for scale in scales if scale not in valid_scales})
    if invalid_scales:
        raise ValueError(
            "Unsupported x-axis scale(s): " + ", ".join(invalid_scales)
        )

    output_paths: Dict[str, Dict[str, Path | None]] = {}
    for category, subset in categories.items():
        label = category_labels.get(category, category)
        filename_template = output_name_template

        if subset.empty:
            print(f"No data for category '{category}', skipping plot")
            output_paths[category] = {scale: None for scale in scales}
            continue

        paths_for_scales: Dict[str, Path | None] = {}
        for scale in scales:
            suffix = "" if scale == "linear" else f"_{scale}"
            paths_for_scales[scale] = _plot_category_violin(
                subset,
                config_col=config_col,
                normalized_col=normalized_col,
                category_key=category,
                category_label=label,
                base_title=base_title,
                output_dir=output_dir,
                output_name_template=filename_template,
                x_scale=scale,
                output_suffix=suffix,
            )

        output_paths[category] = paths_for_scales

    return output_paths


if __name__ == "__main__":
    runs_path = Path("inputs/runs.csv")
    df = pd.read_csv(runs_path)

    df_converged = df[df["is_converged"] == 1].copy()

    print("Generating normalized iteration violin plots by condition category...")
    outputs = plot_normalized_iterations_by_condition_category(df_converged)

    print("\nSummary:")
    for category, scale_paths in outputs.items():
        for scale, path in scale_paths.items():
            label = f"{category} ({scale})"
            if path:
                print(f"  {label}: {path}")
            else:
                print(f"  {label}: No plot generated")
