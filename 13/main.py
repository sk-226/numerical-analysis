from utils.condition_numbers import prepare_runs_dataframe
from plots.convergence_vs_condition import plot_convergence_vs_condition_matplotlib

if __name__ == "__main__":
    df = prepare_runs_dataframe()
    plot_convergence_vs_condition_matplotlib(df)
