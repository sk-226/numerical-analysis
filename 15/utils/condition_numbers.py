from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def load_condition_numbers(condition_path: PathLike) -> pd.DataFrame:
    """Load matrix condition numbers with column validation."""
    condition_df = pd.read_csv(condition_path)
    expected_columns = {"matrix_name", "condition_number"}

    missing_columns = expected_columns - set(condition_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Condition number file {condition_path} is missing columns: {missing_list}"
        )

    return condition_df[["matrix_name", "condition_number"]]


def merge_condition_numbers(
    runs_df: pd.DataFrame, condition_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge condition numbers into solver runs using matrix_name as the key."""
    merged = runs_df.merge(
        condition_df, on="matrix_name", how="left", validate="many_to_one"
    )

    if merged["condition_number"].isna().any():
        missing = merged[merged["condition_number"].isna()]["matrix_name"].unique()
        missing_preview = ", ".join(sorted(missing[:5]))
        print(
            "Warning: missing condition_number for matrices: "
            f"{missing_preview}{'...' if len(missing) > 5 else ''}"
        )

    return merged


def prepare_runs_dataframe(
    runs_path: PathLike = "inputs/runs.csv",
    condition_path: PathLike = "inputs/cond_spd_0-50000.csv",
) -> pd.DataFrame:
    """Load solver runs and attach corresponding condition numbers."""
    runs_df = pd.read_csv(runs_path)
    condition_df = load_condition_numbers(condition_path)
    merged_df = merge_condition_numbers(runs_df, condition_df)

    if "condition_number" not in runs_df.columns:
        merged_df.to_csv(runs_path, index=False)

    return merged_df
