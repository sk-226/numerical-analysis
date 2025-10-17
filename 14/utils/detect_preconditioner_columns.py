import pandas as pd
from typing import Dict

def detect_preconditioner_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect preconditioner-related columns in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        Dict[str, str]
            Dictionary with detected column names
        """
        column_map = {}

        # Check for preconditioner config column
        if "prec_label" in df.columns:
            column_map["config"] = "prec_label"
        elif "preconditioner_configs" in df.columns:
            column_map["config"] = "preconditioner_configs"
        else:
            column_map["config"] = None

        # Check for other important columns
        for col_type, possible_names in [
            ("matrix_name", ["matrix_name", "matrix", "problem"]),
            ("condition_number", ["condition_number", "cond_num", "condition"]),
            ("is_converged", ["is_converged", "converged", "success"]),
            ("iters", ["iters", "iterations", "iter"]),
            ("max_iters", ["max_iters", "max_iterations"]),
            ("matrix_size", ["matrix_size", "size", "n"]),
            ("solve_time", ["solve_time", "time", "runtime"]),
            ("problem_kind", ["problem_kind", "kind", "type"]),
        ]:
            for name in possible_names:
                if name in df.columns:
                    column_map[col_type] = name
                    break
            else:
                column_map[col_type] = None

        return column_map
