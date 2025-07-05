import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, pd, plt


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
    print(table['Converged'].value_counts())
    return


@app.cell
def _(table):
    convergence_by_preconditioner = table.groupby('Preconditioner')['Converged'].value_counts().unstack(fill_value=0)

    # Handle missing columns gracefully
    expected_cols = [-1, 0, 1]
    for col in expected_cols:
        if col not in convergence_by_preconditioner.columns:
            convergence_by_preconditioner[col] = 0

    # Reorder columns to match expected order
    convergence_by_preconditioner = convergence_by_preconditioner.reindex(columns=expected_cols, fill_value=0)

    # Rename columns after ensuring all expected columns exist
    column_mapping = {-1: 'Failed at preconditioning', 0: 'Not converged', 1: 'Converged'}
    convergence_by_preconditioner.rename(columns=column_mapping, inplace=True)

    convergence_by_preconditioner['Total'] = convergence_by_preconditioner.sum(axis=1)
    convergence_by_preconditioner['Success Rate (%)'] = (convergence_by_preconditioner['Converged'] / convergence_by_preconditioner['Total'] * 100).round(1)

    print("Convergence Analysis by Preconditioner:")
    print(convergence_by_preconditioner)
    return (convergence_by_preconditioner,)


@app.cell
def _(table):
    converged_data = table[table['Converged'] == 1].copy()
    if len(converged_data) > 0:
        performance_stats = converged_data.groupby('Preconditioner')[['Iterations', 'SolveTime', 'PreconTime']].agg(['mean', 'median', 'std']).round(6)
        print("Performance Statistics for Successfully Converged Cases:")
        print(performance_stats)
    else:
        print("No successfully converged cases found")
    return (converged_data,)


@app.cell
def _(converged_data, convergence_by_preconditioner, plt):
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Success rate chart
    success_rates = convergence_by_preconditioner['Success Rate (%)']
    ax1.bar(success_rates.index, success_rates.values)
    ax1.set_title('Success Rate by Preconditioner')
    ax1.set_ylabel('Success Rate (%)')
    ax1.tick_params(axis='x', rotation=45)

    # Iterations boxplot
    if len(converged_data) > 0:
        converged_data.boxplot(column='Iterations', by='Preconditioner', ax=ax2)
        ax2.set_title('Iterations Distribution (Converged Cases)')
        ax2.set_xlabel('Preconditioner')
        ax2.set_ylabel('Iterations')

        # Solve time boxplot
        converged_data.boxplot(column='SolveTime', by='Preconditioner', ax=ax3)
        ax3.set_title('Solve Time Distribution (Converged Cases)')
        ax3.set_xlabel('Preconditioner')
        ax3.set_ylabel('Solve Time (s)')

        # Preconditioning time boxplot
        converged_data.boxplot(column='PreconTime', by='Preconditioner', ax=ax4)
        ax4.set_title('Preconditioning Time Distribution')
        ax4.set_xlabel('Preconditioner')
        ax4.set_ylabel('Preconditioning Time (s)')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(table):
    print("Summary of Key Findings:")
    print("="*50)
    total_cases = len(table)

    # Overall convergence
    overall_converged = len(table[table['Converged'] == 1])
    overall_failed_precon = len(table[table['Converged'] == -1])
    overall_not_converged = len(table[table['Converged'] == 0])

    print(f"Total test cases: {total_cases}")
    print(f"Successfully converged: {overall_converged} ({overall_converged/total_cases*100:.1f}%)")
    print(f"Failed at preconditioning: {overall_failed_precon} ({overall_failed_precon/total_cases*100:.1f}%)")
    print(f"Did not converge: {overall_not_converged} ({overall_not_converged/total_cases*100:.1f}%)")

    # Best performing preconditioner
    if overall_converged > 0:
        converged_only = table[table['Converged'] == 1]
        avg_iterations = converged_only.groupby('Preconditioner')['Iterations'].mean()
        best_preconditioner = avg_iterations.idxmin()
        print(f"\nBest preconditioner (fewest iterations): {best_preconditioner}")
        print(f"Average iterations: {avg_iterations[best_preconditioner]:.1f}")

        # Most reliable preconditioner 
        reliability = table.groupby('Preconditioner')['Converged'].apply(lambda x: (x == 1).mean() * 100)
        most_reliable = reliability.idxmax()
        print(f"\nMost reliable preconditioner: {most_reliable}")
        print(f"Success rate: {reliability[most_reliable]:.1f}%")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 

    - success rate をパット見 SSOR のほうがよく見えはするが, ic は ichol で失敗している場合がかなりあると思う.
    """
    )
    return


@app.cell
def _(table):
    # IC前処理器の詳細分析
    ic_analysis = {}
    ic_analysis['data'] = table[table['Preconditioner'] == 'ic'].copy()

    print("IC前処理器の詳細分析:")
    print("="*50)

    # ICでの失敗判定 (Converged=0 and Iterations=-1)
    ic_analysis['failed_ichol'] = ic_analysis['data'][(ic_analysis['data']['Converged'] == 0) & (ic_analysis['data']['Iterations'] == -1)]
    ic_analysis['converged'] = ic_analysis['data'][ic_analysis['data']['Converged'] == 1]
    ic_analysis['not_converged'] = ic_analysis['data'][(ic_analysis['data']['Converged'] == 0) & (ic_analysis['data']['Iterations'] != -1)]

    ic_analysis['stats'] = {
        'total': len(ic_analysis['data']),
        'failed_ichol_count': len(ic_analysis['failed_ichol']),
        'converged_count': len(ic_analysis['converged']),
        'not_converged_count': len(ic_analysis['not_converged'])
    }

    stats = ic_analysis['stats']
    print(f"IC前処理器の総テストケース数: {stats['total']}")
    print(f"Ichol失敗: {stats['failed_ichol_count']} ({stats['failed_ichol_count']/stats['total']*100:.1f}%)")
    print(f"収束成功: {stats['converged_count']} ({stats['converged_count']/stats['total']*100:.1f}%)")
    print(f"収束失敗: {stats['not_converged_count']} ({stats['not_converged_count']/stats['total']*100:.1f}%)")

    print(f"\nIchol失敗した行列リスト ({stats['failed_ichol_count']}個):")
    print("-" * 40)
    ic_analysis['failed_matrices'] = ic_analysis['failed_ichol']['Matrix'].tolist()
    for i, matrix in enumerate(ic_analysis['failed_matrices'], 1):
        print(f"{i:2d}. {matrix}")

    return (ic_analysis,)


@app.cell
def _(ic_analysis, table):
    # SSORとICの比較分析
    print("SSORとICの比較:")
    print("="*50)

    ssor_data = table[table['Preconditioner'] == 'ssor']
    ssor_stats = {
        'total': len(ssor_data),
        'converged': len(ssor_data[ssor_data['Converged'] == 1])
    }

    ic_stats = ic_analysis['stats']

    print(f"SSOR成功率: {ssor_stats['converged']/ssor_stats['total']*100:.1f}% ({ssor_stats['converged']}/{ssor_stats['total']})")
    print(f"IC全体成功率: {ic_stats['converged_count']/ic_stats['total']*100:.1f}% ({ic_stats['converged_count']}/{ic_stats['total']})")

    # ICでicholが成功した場合の成功率
    ic_ichol_success = ic_analysis['data'][ic_analysis['data']['Iterations'] != -1]  # ichol成功ケース
    ic_ichol_success_stats = {
        'converged': len(ic_ichol_success[ic_ichol_success['Converged'] == 1]),
        'total': len(ic_ichol_success)
    }

    if ic_ichol_success_stats['total'] > 0:
        success_rate = ic_ichol_success_stats['converged']/ic_ichol_success_stats['total']*100
        print(f"IC(ichol成功時)の成功率: {success_rate:.1f}% ({ic_ichol_success_stats['converged']}/{ic_ichol_success_stats['total']})")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## NOTE

    - ICはichol前処理が成功すれば100%収束する
    - ICの問題点: ただし, 約40.7%の行列でichol自体が失敗する. プログラム的に ichol が失敗することはすぐわかるので good.
        - ただし, 多重で行うとき, for の中で 前処理が失敗するとシンプルにだるい. (エラーハンドリングという意味でも)
    - SSORとの比較として, SSORは90.7%の安定した成功率を示すが、ICがichol成功時には100%の成功率を達成している
    - 何も考えずに 全体の 90.7% が成功する SSOR は遅いどうこうはあるが, 結構すごいのでは？
        - まあ, ic は ichol そのものが成功すれば 100% なので... 感はある.

    ⚠️ 時間というより, 行列サイズ n に対して, どれだけの反復回数で収束したかを check すべきだった.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Residual, Error""")
    return


if __name__ == "__main__":
    app.run()
