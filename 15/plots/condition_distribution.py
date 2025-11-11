from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.detect_preconditioner_columns import detect_preconditioner_columns


def load_unique_condition_numbers(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    column_map = detect_preconditioner_columns(df)
    matrix_col = column_map.get("matrix_name")
    condition_col = column_map.get("condition_number")
    if not matrix_col or not condition_col:
        raise ValueError(
            "Required columns 'matrix_name' and 'condition_number' not found"
        )

    working = pd.DataFrame(
        {
            "matrix_name": df[matrix_col],
            "condition_number": pd.to_numeric(df[condition_col], errors="coerce"),
        }
    )
    working = working.dropna(subset=["matrix_name", "condition_number"])
    working = working[working["condition_number"] > 0]

    if working.empty:
        raise ValueError("No positive condition numbers available")

    grouped = working.groupby("matrix_name")["condition_number"].median()
    grouped.name = "condition_number"
    return grouped


def plot_condition_distribution(
    condition_numbers: pd.Series, output_path: Path
) -> None:
    data = np.log10(condition_numbers.to_numpy())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=40, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.set_xlabel("log10(Condition Number)")
    ax.set_ylabel("Number of Matrices")
    ax.set_title("Condition Number Distribution")
    ax.grid(axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format=output_path.suffix.lstrip("."))
    plt.close(fig)


if __name__ == "__main__":
    runs_path = Path("inputs/runs.csv")
    output_path = Path("outputs/condition_number_distribution.svg")
    try:
        condition_numbers = load_unique_condition_numbers(runs_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Unable to produce distribution: {exc}")
        sys.exit(1)

    plot_condition_distribution(condition_numbers, output_path)
    print(f"Plot saved to {output_path}")
