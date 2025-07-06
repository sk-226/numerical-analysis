import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np

    import warnings

    warnings.filterwarnings("ignore")

    return go, mo, np, pd


@app.cell
def _(pd):
    table = pd.read_csv("outputs/exp_cmp_preconditioner/results_comp_prec.csv")
    return (table,)


@app.cell
def _(table):
    table
    return


@app.cell
def _(table):
    print(f"Dataset shape: {table.shape}")
    print(f"Columns: {list(table.columns)}")
    print(f"\nPreconditioner types: {table['Preconditioner'].unique()}")
    print(f"\nConvergence status distribution:")
    print(table["Converged"].value_counts())
    return


@app.cell
def _(mo):
    mo.md(r"""# Convergence""")
    return


@app.cell
def _(pd, table):
    # Create convergence analysis with proper categorization
    result_data = []
    for prec in table["Preconditioner"].unique():
        prec_data = table[table["Preconditioner"] == prec]

        converged = len(prec_data[prec_data["Converged"] == 1])
        failed_precon = len(prec_data[prec_data["Iterations"] == -1])
        not_converged = len(
            prec_data[(prec_data["Converged"] == 0) & (prec_data["Iterations"] != -1)]
        )

        result_data.append(
            {
                "Preconditioner": prec,
                "Failed at preconditioning": failed_precon,
                "Not converged": not_converged,
                "Converged": converged,
            }
        )

    convergence_by_preconditioner = pd.DataFrame(result_data).set_index(
        "Preconditioner"
    )

    convergence_by_preconditioner["Total"] = convergence_by_preconditioner.sum(axis=1)
    convergence_by_preconditioner["Success Rate (%)"] = (
        convergence_by_preconditioner["Converged"]
        / convergence_by_preconditioner["Total"]
        * 100
    ).round(1)

    print("Convergence Analysis by Preconditioner:")
    print(convergence_by_preconditioner)
    return (convergence_by_preconditioner,)


@app.cell
def _(table):
    converged_data = table[table["Converged"] == 1].copy()
    if len(converged_data) > 0:
        performance_stats = (
            converged_data.groupby("Preconditioner")[
                ["Iterations", "SolveTime", "PreconTime"]
            ]
            .agg(["mean", "median", "std"])
            .round(6)
        )
        print("Performance Statistics for Successfully Converged Cases:")
        print(performance_stats)
    else:
        print("No successfully converged cases found")
    return (converged_data,)


@app.cell
def _(convergence_by_preconditioner, go):
    # Success rate chart
    success_rates = convergence_by_preconditioner["Success Rate (%)"]

    success_fig = go.Figure(
        data=[
            go.Bar(
                x=list(success_rates.index),
                y=list(success_rates.values),
                name="Success Rate",
                marker_color="steelblue",
            )
        ]
    )

    success_fig.update_layout(
        title="Success Rate by Preconditioner",
        xaxis_title="Preconditioner",
        yaxis_title="Success Rate (%)",
    )

    return


@app.cell
def _(table):
    print("Summary of Key Findings:")
    print("=" * 50)
    total_cases = len(table)

    # Overall convergence
    overall_converged = len(table[table["Converged"] == 1])
    overall_failed_precon = len(
        table[table["Iterations"] == -1]
    )  # Preconditioning failure detected by Iterations=-1
    overall_not_converged = len(
        table[(table["Converged"] == 0) & (table["Iterations"] != -1)]
    )  # Non-convergence (preconditioner succeeded)

    print(f"Total test cases: {total_cases}")
    print(
        f"Successfully converged: {overall_converged} ({overall_converged / total_cases * 100:.1f}%)"
    )
    print(
        f"Failed at preconditioning: {overall_failed_precon} ({overall_failed_precon / total_cases * 100:.1f}%)"
    )
    print(
        f"Did not converge: {overall_not_converged} ({overall_not_converged / total_cases * 100:.1f}%)"
    )

    # Best performing preconditioner
    if overall_converged > 0:
        converged_only = table[table["Converged"] == 1]
        avg_iterations = converged_only.groupby("Preconditioner")["Iterations"].mean()
        best_preconditioner = avg_iterations.idxmin()
        print(f"\nBest preconditioner (fewest iterations): {best_preconditioner}")
        print(f"Average iterations: {avg_iterations[best_preconditioner]:.1f}")

        # Most reliable preconditioner?
        def calculate_reliability(group):
            converged = (group["Converged"] == 1).sum()
            total = len(group)
            return (converged / total) * 100

        reliability = table.groupby("Preconditioner").apply(calculate_reliability)
        most_reliable = reliability.idxmax()
        print(f"\nMost reliable preconditioner: {most_reliable}")
        print(f"Success rate: {reliability[most_reliable]:.1f}%")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## NOTE

    - While SSOR appears to have better success rates at first glance, IC fails frequently due to ichol failures.
    - These are just success counts, not quality metrics 👈
    """
    )
    return


@app.cell
def _(table):
    # Detailed analysis of IC preconditioner
    ic_analysis = {}
    ic_analysis["data"] = table[table["Preconditioner"] == "ic"].copy()

    print("Detailed Analysis of IC Preconditioner:")
    print("=" * 50)

    # IC failure detection (Converged=0 and Iterations=-1)
    ic_analysis["failed_ichol"] = ic_analysis["data"][
        (ic_analysis["data"]["Converged"] == 0)
        & (ic_analysis["data"]["Iterations"] == -1)
    ]
    ic_analysis["converged"] = ic_analysis["data"][
        ic_analysis["data"]["Converged"] == 1
    ]
    ic_analysis["not_converged"] = ic_analysis["data"][
        (ic_analysis["data"]["Converged"] == 0)
        & (ic_analysis["data"]["Iterations"] != -1)
    ]

    ic_analysis["stats"] = {
        "total": len(ic_analysis["data"]),
        "failed_ichol_count": len(ic_analysis["failed_ichol"]),
        "converged_count": len(ic_analysis["converged"]),
        "not_converged_count": len(ic_analysis["not_converged"]),
    }

    stats = ic_analysis["stats"]
    print(f"Total IC preconditioner test cases: {stats['total']}")
    print(
        f"Ichol failures: {stats['failed_ichol_count']} ({stats['failed_ichol_count'] / stats['total'] * 100:.1f}%)"
    )
    print(
        f"Convergence success: {stats['converged_count']} ({stats['converged_count'] / stats['total'] * 100:.1f}%)"
    )
    print(
        f"Convergence failure: {stats['not_converged_count']} ({stats['not_converged_count'] / stats['total'] * 100:.1f}%)"
    )

    print(f"\nMatrices with Ichol failures ({stats['failed_ichol_count']} cases):")
    print("-" * 40)
    ic_analysis["failed_matrices"] = ic_analysis["failed_ichol"]["Matrix"].tolist()
    for i, matrix_name in enumerate(ic_analysis["failed_matrices"], 1):
        print(f"{i:2d}. {matrix_name}")

    return (ic_analysis,)


@app.cell
def _(ic_analysis, table):
    # Comparison between SSOR and IC preconditioners
    print("SSOR vs IC Comparison:")
    print("=" * 50)

    ssor_data = table[table["Preconditioner"] == "ssor"]
    ssor_stats = {
        "total": len(ssor_data),
        "converged": len(ssor_data[ssor_data["Converged"] == 1]),
    }

    ic_stats = ic_analysis["stats"]

    print(
        f"SSOR success rate: {ssor_stats['converged'] / ssor_stats['total'] * 100:.1f}% ({ssor_stats['converged']}/{ssor_stats['total']})"
    )
    print(
        f"IC overall success rate: {ic_stats['converged_count'] / ic_stats['total'] * 100:.1f}% ({ic_stats['converged_count']}/{ic_stats['total']})"
    )

    # Success rate when IC's ichol succeeds
    ic_ichol_success = ic_analysis["data"][
        ic_analysis["data"]["Iterations"] != -1
    ]  # Cases where ichol succeeded
    ic_ichol_success_stats = {
        "converged": len(ic_ichol_success[ic_ichol_success["Converged"] == 1]),
        "total": len(ic_ichol_success),
    }

    if ic_ichol_success_stats["total"] > 0:
        success_rate = (
            ic_ichol_success_stats["converged"] / ic_ichol_success_stats["total"] * 100
        )
        print(
            f"IC success rate (when ichol succeeds): {success_rate:.1f}% ({ic_ichol_success_stats['converged']}/{ic_ichol_success_stats['total']})"
        )

    return


@app.cell
def _(converged_data, go, np):
    # Matrix size vs iterations scatter plot
    size_iter_fig = go.Figure()

    # Color-coded plot by preconditioner type
    colors = ["blue", "red", "green", "orange", "purple"]

    for idx, preconditioner_name in enumerate(
        converged_data["Preconditioner"].unique()
    ):
        size_prec_data = converged_data[
            converged_data["Preconditioner"] == preconditioner_name
        ]

        size_iter_fig.add_trace(
            go.Scatter(
                x=size_prec_data["MatrixSize"].values,
                y=size_prec_data["Iterations"].values,
                mode="markers",
                name=preconditioner_name,
                marker=dict(size=6, opacity=0.7, color=colors[idx % len(colors)]),
                text=size_prec_data["Matrix"].values,  # Matrix names
                customdata=np.column_stack(
                    [
                        size_prec_data["MatrixSize"].values,  # Matrix size
                        size_prec_data["Iterations"].values,  # Iteration count
                        size_prec_data["SolveTime"].values,  # Solve time
                        size_prec_data["PreconTime"].values,  # Preconditioning time
                    ]
                ),
                hovertemplate="<b>%{text}</b><br>"
                + "Preconditioner: "
                + preconditioner_name
                + "<br>"
                + "Matrix Size: %{customdata[0]}<br>"
                + "Iterations: %{customdata[1]}<br>"
                + "Solve Time: %{customdata[2]:.4f}s<br>"
                + "Precon Time: %{customdata[3]:.4f}s<extra></extra>",
            )
        )

    size_iter_fig.update_layout(
        title="Matrix Size vs Iterations (Converged Cases)",
        xaxis_title="Matrix Size",
        yaxis_title="Iterations",
        xaxis_type="log",
        yaxis_type="log",
        width=900,
        height=600,
    )


    return


@app.cell
def _(converged_data, go, np):
    # Analysis of iterations/matrix size ratio by preconditioner
    efficiency_fig = go.Figure()

    # Calculate iterations/matrix_size for each converged case
    converged_data_with_ratio = converged_data.copy()
    converged_data_with_ratio["IterPerSize"] = (
        converged_data_with_ratio["Iterations"]
        / converged_data_with_ratio["MatrixSize"]
    )

    # Display as box plot by preconditioner type
    for precond_name in converged_data_with_ratio["Preconditioner"].unique():
        precond_data = converged_data_with_ratio[
            converged_data_with_ratio["Preconditioner"] == precond_name
        ]

        efficiency_fig.add_trace(
            go.Box(
                y=precond_data["IterPerSize"].values,
                name=precond_name,
                boxpoints="outliers",
                text=precond_data["Matrix"].values,
                customdata=np.column_stack(
                    [
                        precond_data["MatrixSize"].values,
                        precond_data["Iterations"].values,
                        precond_data["IterPerSize"].values,
                    ]
                ),
                hovertemplate="<b>%{text}</b><br>"
                + "Preconditioner: "
                + precond_name
                + "<br>"
                + "Matrix Size: %{customdata[0]}<br>"
                + "Iterations: %{customdata[1]}<br>"
                + "Iter/Size: %{customdata[2]:.4f}<extra></extra>",
            )
        )

    efficiency_fig.update_layout(
        title="Preconditioner Efficiency (Iterations/Matrix Size)",
        xaxis_title="Preconditioner",
        yaxis_title="Iterations per Matrix Size",
        yaxis_type="log",
        width=900,
        height=600,
    )

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## NOTE

    - IC achieves 100% convergence when ichol preconditioning succeeds
    - IC's problem: ichol itself fails for ~40.7% of matrices. Programmatically, ichol failures are immediately detectable (good).
        - However, when processing multiple matrices in a loop, preconditioner failures create error handling complexity
    - SSOR vs IC comparison: SSOR shows stable 90.7% success rate, while IC achieves 100% success when ichol succeeds
    - SSOR's 90.7% success rate without any special consideration is quite impressive, despite potential performance issues
        - Though IC is 100% when ichol succeeds, that's a conditional success...
    """
    )
    return


@app.cell
def _(converged_data, go):
    # Iterations distribution by preconditioner
    iterations_fig = go.Figure()

    if len(converged_data) > 0:
        for prec_iter in converged_data["Preconditioner"].unique():
            iter_values = converged_data[converged_data["Preconditioner"] == prec_iter][
                "Iterations"
            ].values
            iterations_fig.add_trace(
                go.Box(y=iter_values, name=prec_iter, boxpoints="outliers")
            )

    iterations_fig.update_layout(
        title="Iterations Distribution (Converged Cases)",
        xaxis_title="Preconditioner",
        yaxis_title="Iterations",
    )

    return


@app.cell
def _(converged_data, go):
    # Solve time distribution by preconditioner
    solvetime_fig = go.Figure()

    if len(converged_data) > 0:
        for prec_time in converged_data["Preconditioner"].unique():
            time_values = converged_data[converged_data["Preconditioner"] == prec_time][
                "SolveTime"
            ].values
            solvetime_fig.add_trace(
                go.Box(y=time_values, name=prec_time, boxpoints="outliers")
            )

    solvetime_fig.update_layout(
        title="Solve Time Distribution (Converged Cases)",
        xaxis_title="Preconditioner",
        yaxis_title="Solve Time (s)",
        yaxis_type="log",
    )

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Residual, Error Analysis

    TODO: residual gap について, 差にするべきだった
    """
    )
    return


@app.cell
def _(np, table):
    # Extract converged cases only for error analysis
    converged_cases = table[table["Converged"] == 1].copy()

    print("Error Metrics Analysis for Successfully Converged Cases:")
    print("=" * 50)
    print(f"Analysis target cases: {len(converged_cases)}")

    # Residual gap analysis (zero division protection)
    # Exclude cases with very small FinalRelRes2 values (< 1e-16)
    valid_cases = converged_cases[converged_cases["FinalRelRes2"] >= 1e-16].copy()
    valid_cases["ResidualGap"] = (
        valid_cases["TrueRelRes2"] / valid_cases["FinalRelRes2"]
    )
    valid_cases["ResidualGapDigits"] = np.log10(valid_cases["ResidualGap"])

    print(
        f"\nValid cases: {len(valid_cases)} / {len(converged_cases)} (excluded: {len(converged_cases) - len(valid_cases)} cases)"
    )

    print("\nResidual gap statistics:")
    print(f"Gap mean: {valid_cases['ResidualGap'].mean():.2f}")
    print(f"Gap median: {valid_cases['ResidualGap'].median():.2f}")
    print(f"Gap maximum: {valid_cases['ResidualGap'].max():.2e}")
    print(f"Gap digits mean: {valid_cases['ResidualGapDigits'].mean():.2f} digits")
    print(f"Gap digits median: {valid_cases['ResidualGapDigits'].median():.2f} digits")
    print(f"Gap digits maximum: {valid_cases['ResidualGapDigits'].max():.2f} digits")
    print(f"Cases with residual gap > 1 digit: {(valid_cases['ResidualGapDigits'] > 1).sum()}")
    print(f"Cases with residual gap > 2 digits: {(valid_cases['ResidualGapDigits'] > 2).sum()}")

    return (valid_cases,)


@app.cell
def _(valid_cases):
    # Residual gap analysis by preconditioner
    print("Residual gap analysis by preconditioner:")
    print("=" * 50)

    gap_stats = (
        valid_cases.groupby("Preconditioner")["ResidualGap"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(3)
    )

    print("Preconditioner-wise Residual gap statistics (count/mean/median/std/min/max):")
    print(gap_stats)

    # Analysis of large residual gap cases by preconditioner
    print("\nLarge Residual Gap (>1 digit) occurrence rate by preconditioner:")
    large_gap_by_prec = (
        valid_cases.groupby("Preconditioner")
        .apply(lambda x: (x["ResidualGapDigits"] > 1).sum() / len(x) * 100)
        .round(1)
    )

    for prec_gap_name, rate in large_gap_by_prec.items():
        count = (
            valid_cases[valid_cases["Preconditioner"] == prec_gap_name][
                "ResidualGapDigits"
            ]
            > 1
        ).sum()
        total_cases_for_prec = len(
            valid_cases[valid_cases["Preconditioner"] == prec_gap_name]
        )
        print(f"  {prec_gap_name}: {rate}% ({count}/{total_cases_for_prec})")

    return


@app.cell
def _(valid_cases):
    # True residual < tol achievement analysis
    print("True residual < tol Achievement Analysis:")
    print("=" * 50)

    # True residual < tol vs formal convergence
    formal_convergence = len(valid_cases)
    true_high_precision = (valid_cases["TrueRelRes2"] <= 1e-12).sum()

    print(f"Valid cases: {formal_convergence}")
    print(
        f"TrueRelRes2 <= 1e-12 achieved: {true_high_precision} ({true_high_precision / formal_convergence * 100:.1f}%)"
    )

    # (true residual < tol) achievement rate by preconditioner
    print("\n(True residual < tol) achievement rate by preconditioner:")
    true_precision_by_prec = (
        valid_cases.groupby("Preconditioner")
        .apply(lambda x: (x["TrueRelRes2"] <= 1e-12).sum() / len(x) * 100)
        .round(1)
    )

    for preconditioner, precision_rate in true_precision_by_prec.items():
        achieved = (
            valid_cases[valid_cases["Preconditioner"] == preconditioner]["TrueRelRes2"]
            <= 1e-12
        ).sum()
        total_cases_for_precision = len(
            valid_cases[valid_cases["Preconditioner"] == preconditioner]
        )
        print(
            f"  {preconditioner}: {precision_rate}% ({achieved}/{total_cases_for_precision})"
        )

    return


@app.cell
def _(go, valid_cases):
    # NOTE:
    #     valid_cases <- is_converged=true;
    #     valid_cases['TrueRelRes2'] / valid_cases['FinalRelRes2']
    # Gap distribution by preconditioner (Box Plot)
    gap_box_fig = go.Figure()

    preconditioners = valid_cases["Preconditioner"].unique()
    for prec_gap in preconditioners:
        gap_values = valid_cases[valid_cases["Preconditioner"] == prec_gap][
            "ResidualGap"
        ].values
        gap_box_fig.add_trace(go.Box(y=gap_values, name=prec_gap, boxpoints="outliers"))

    gap_box_fig.update_layout(
        title="Residual Gap by Preconditioner",
        xaxis_title="Preconditioner",
        yaxis_title="Residual Gap",
        yaxis_type="log",
    )

    return


@app.cell
def _(go, np, valid_cases):
    # True vs Final Residual scatter plot
    residual_fig = go.Figure()

    # Color-coded plot by preconditioner type
    residual_colors = ["blue", "red", "green", "orange", "purple"]

    for residual_idx, residual_prec_name in enumerate(
        valid_cases["Preconditioner"].unique()
    ):
        residual_prec_data = valid_cases[
            valid_cases["Preconditioner"] == residual_prec_name
        ]

        residual_fig.add_trace(
            go.Scatter(
                x=np.log10(residual_prec_data["FinalRelRes2"].values),
                y=np.log10(residual_prec_data["TrueRelRes2"].values),
                mode="markers",
                name=residual_prec_name,
                marker=dict(
                    size=4,
                    opacity=0.6,
                    color=residual_colors[residual_idx % len(residual_colors)],
                ),
                text=residual_prec_data["Matrix"].values,
                customdata=np.column_stack(
                    [
                        residual_prec_data["Iterations"].values,
                        residual_prec_data["FinalRelRes2"].values,
                        residual_prec_data["TrueRelRes2"].values,
                    ]
                ),
                hovertemplate="<b>%{text}</b><br>"
                + "Preconditioner: "
                + residual_prec_name
                + "<br>"
                + "Iterations: %{customdata[0]}<br>"
                + "log₁₀(Final): %{x:.2f}<br>"
                + "log₁₀(True): %{y:.2f}<br>"
                + "Final Residual: %{customdata[1]:.2e}<br>"
                + "True Residual: %{customdata[2]:.2e}<extra></extra>",
            )
        )

    # y=x line (log10 scale)
    residual_fig.add_trace(
        go.Scatter(
            x=[np.log10(1e-16), np.log10(1e-6)],
            y=[np.log10(1e-16), np.log10(1e-6)],
            mode="lines",
            name="y=x",
            line=dict(dash="dash", color="lightgray", width=1),
            hoverinfo="skip",  # Skip hover info for y=x line
        )
    )

    residual_fig.update_layout(
        title="True vs Final Residual (log₁₀ scale)",
        xaxis_title="log₁₀(Final Relative Residual)",
        yaxis_title="log₁₀(True Relative Residual)",
    )

    return


@app.cell
def _(go, np, valid_cases):
    # 2-norm vs A-norm Error scatter plot
    error_fig = go.Figure()

    # Color-coded plot by preconditioner type
    error_colors = ["blue", "red", "green", "orange", "purple"]

    for error_idx, error_prec_name in enumerate(valid_cases["Preconditioner"].unique()):
        error_prec_data = valid_cases[valid_cases["Preconditioner"] == error_prec_name]

        error_fig.add_trace(
            go.Scatter(
                x=np.log10(error_prec_data["FinalRelErr2"].values),
                y=np.log10(error_prec_data["FinalRelErrA"].values),
                mode="markers",
                name=error_prec_name,
                marker=dict(
                    size=4,
                    opacity=0.6,
                    color=error_colors[error_idx % len(error_colors)],
                ),
                text=error_prec_data["Matrix"].values,
                customdata=np.column_stack(
                    [
                        error_prec_data["Iterations"].values,
                        error_prec_data["FinalRelErr2"].values,
                        error_prec_data["FinalRelErrA"].values,
                    ]
                ),
                hovertemplate="<b>%{text}</b><br>"
                + "Preconditioner: "
                + error_prec_name
                + "<br>"
                + "Iterations: %{customdata[0]}<br>"
                + "log₁₀(2-norm): %{x:.2f}<br>"
                + "log₁₀(A-norm): %{y:.2f}<br>"
                + "2-norm Error: %{customdata[1]:.2e}<br>"
                + "A-norm Error: %{customdata[2]:.2e}<extra></extra>",
            )
        )

    # y=x line (log10 scale)
    error_fig.add_trace(
        go.Scatter(
            x=[np.log10(1e-16), np.log10(1e-6)],
            y=[np.log10(1e-16), np.log10(1e-6)],
            mode="lines",
            name="y=x",
            line=dict(dash="dash", color="lightgray", width=1),
            hoverinfo="skip",  # Skip hover info for y=x line
        )
    )

    error_fig.update_layout(
        title="2-norm vs A-norm Error (log₁₀ scale)",
        xaxis_title="log₁₀(Final Relative Error 2-norm)",
        yaxis_title="log₁₀(Final Relative Error A-norm)",
    )

    return


@app.cell
def _(valid_cases):
    print("Detailed Analysis of Extreme Residual Gap Cases:")
    print("=" * 50)

    # Cases with Gap > 3 digits
    extreme_cases = valid_cases[valid_cases["ResidualGapDigits"] > 3].copy()

    if len(extreme_cases) > 0:
        print(f"Residual Gap > 3 digits cases ({len(extreme_cases)} cases):")
        extreme_cases_display = extreme_cases[
            [
                "Matrix",
                "Preconditioner",
                "ResidualGap",
                "ResidualGapDigits",
                "TrueRelRes2",
                "FinalRelRes2",
                "Iterations",
            ]
        ].copy()
        extreme_cases_display = extreme_cases_display.sort_values(
            "ResidualGapDigits", ascending=False
        )

        for _, row in extreme_cases_display.iterrows():
            print(
                f"  {row['Matrix']} ({row['Preconditioner']}): Gap={row['ResidualGapDigits']:.2f} digits, True={row['TrueRelRes2']:.2e}, Final={row['FinalRelRes2']:.2e}, Iter={row['Iterations']}"
            )
    else:
        print("No Residual Gap > 3 digits cases found")

    # Cases with Gap > 2 digits
    large_gap_cases = valid_cases[valid_cases["ResidualGapDigits"] > 2].copy()

    if len(large_gap_cases) > 0:
        print(f"\nResidual Gap > 2 digits cases: {len(large_gap_cases)}")
        print("Occurrence frequency:")
        matrix_gap_count = large_gap_cases["Matrix"].value_counts()
        for mat_name, freq_count in matrix_gap_count.items():
            print(f"  {mat_name}: {freq_count} times")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
