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
