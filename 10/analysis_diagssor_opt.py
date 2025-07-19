import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from ssdownload import SuiteSparseDownloader, Filter
    from pathlib import Path

    from typing import List


    warnings.filterwarnings("ignore")
    return Filter, List, SuiteSparseDownloader, mo, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(
        r"""
    - diag+ssor の場合緩和係数の値で 2,3 個収束しない場合もあったが, 基本的に収束の振る舞いは同じ.
    - 一方で, 反復の回数はもちろん変わる.
        - 1.0, 1.25, 0.75, 1.5 の順でもっとも反復回数が少ない
        - 反復回数および収束率のどちらに関しても言えば, $\omega = 1.0$ がもっともよい.
            - （0.75 は収束率では同じ, しかし反復回数が少し多い??）
    """
    )
    return


@app.cell
def _(np, plt, sns):
    def plot_convergence_by_preconditioner_config(df):
        """Visualize convergence rates by preconditioner configuration with line plots for each problem kind"""
    
        # 1. Data preprocessing
        # Check for correct column name
        if 'preconditioner_configs' in df.columns:
            config_col = 'preconditioner_configs'
        elif 'preconditioner_config' in df.columns:
            config_col = 'preconditioner_config'
        else:
            print("Warning: preconditioner_config column not found")
            return
    
        # Calculate convergence rates for each combination
        convergence_data = df.groupby(['problem_kind', config_col]).agg({
            'is_converged': ['mean', 'count']
        }).round(4)
    
        convergence_data.columns = ['convergence_rate', 'sample_count']
        convergence_data = convergence_data.reset_index()
    
        # Filter out combinations with insufficient samples (minimum 3 samples)
        convergence_data = convergence_data[convergence_data['sample_count'] >= 3]
    
        print(f"Data Summary:")
        print(f"- Number of problem kinds: {convergence_data['problem_kind'].nunique()}")
        print(f"- Number of preconditioner configs: {convergence_data[config_col].nunique()}")
        print(f"- Valid combinations: {len(convergence_data)}")
    
        # 2. Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
        # Set up color palette for problem kinds
        problem_kinds = convergence_data['problem_kind'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(problem_kinds)))
        color_map = dict(zip(problem_kinds, colors))
    
        # Define line styles and markers for visual separation
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
    
        # 2-1. Main line plot with visual separation techniques
        for i, problem_kind in enumerate(problem_kinds):
            data_subset = convergence_data[convergence_data['problem_kind'] == problem_kind]
        
            # Sort data for proper line connections
            data_subset = data_subset.sort_values(config_col)
        
            # Add small jitter to y-values to separate overlapping lines
            jitter_amount = 0.005  # Small offset
            jitter = np.random.uniform(-jitter_amount, jitter_amount, len(data_subset))
            y_values = data_subset['convergence_rate'] + jitter
        
            # Cycle through line styles and markers
            line_style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
        
            # Vary line width slightly
            line_width = 2 + (i % 3) * 0.5
        
            ax1.plot(data_subset[config_col], y_values, 
                    linestyle=line_style, marker=marker, label=problem_kind, 
                    color=color_map[problem_kind], linewidth=line_width, 
                    markersize=6, alpha=0.85, markerfacecolor='white', 
                    markeredgewidth=1.5)
    
        ax1.set_xlabel('Preconditioner Configuration', fontsize=12)
        ax1.set_ylabel('Convergence Rate', fontsize=12)
        ax1.set_title('Convergence Rate by Preconditioner Configuration and Problem Kind', 
                      fontsize=14, fontweight='bold')
        ax1.set_ylim(-0.02, 1.08)  # Extended range to show separated lines better
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
        # Legend settings with better layout for many items
        if len(problem_kinds) > 10:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=9)
        else:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=10)
    
        # 2-2. Sample count heatmap for validation
        sample_counts = df.groupby(['problem_kind', config_col]).size().reset_index(name='count')
        sample_pivot = sample_counts.pivot(index='problem_kind', columns=config_col, values='count').fillna(0)
    
        sns.heatmap(sample_pivot, annot=True, fmt='g', cmap='Blues', 
                    ax=ax2, cbar_kws={'label': 'Sample Count'})
        ax2.set_title('Sample Count by Problem Kind and Preconditioner Configuration', 
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Preconditioner Configuration')
        ax2.set_ylabel('Problem Kind')
    
        plt.tight_layout()
        plt.show()
    
        # 3. Statistical summary
        print("\n=== Statistics by Preconditioner Configuration ===")
        config_stats = convergence_data.groupby(config_col).agg({
            'convergence_rate': ['mean', 'std', 'min', 'max'],
            'sample_count': 'sum'
        }).round(4)
        config_stats.columns = ['avg_convergence', 'std_convergence', 'min_convergence', 
                               'max_convergence', 'total_samples']
        config_stats = config_stats.reset_index().sort_values('avg_convergence', ascending=False)
        print(config_stats)
    
        print("\n=== Statistics by Problem Kind ===")
        problem_stats = convergence_data.groupby('problem_kind').agg({
            'convergence_rate': ['mean', 'std', 'min', 'max'],
            'sample_count': 'sum'
        }).round(4)
        problem_stats.columns = ['avg_convergence', 'std_convergence', 'min_convergence', 
                                'max_convergence', 'total_samples']
        problem_stats = problem_stats.reset_index().sort_values('avg_convergence', ascending=False)
        print(problem_stats)
    
        # 4. Extract notable combinations
        print("\n=== Notable Combinations ===")
    
        # Best convergence combination
        best_combo = convergence_data.loc[convergence_data['convergence_rate'].idxmax()]
        print(f"Best convergence: {best_combo['problem_kind']} + {best_combo[config_col]} = {best_combo['convergence_rate']:.4f}")
    
        # Worst convergence combination
        worst_combo = convergence_data.loc[convergence_data['convergence_rate'].idxmin()]
        print(f"Worst convergence: {worst_combo['problem_kind']} + {worst_combo[config_col]} = {worst_combo['convergence_rate']:.4f}")
    
        # Most effective problem kind for each preconditioner config
        print("\nMost effective problem kind for each preconditioner config:")
        for config in convergence_data[config_col].unique():
            config_data = convergence_data[convergence_data[config_col] == config]
            if len(config_data) > 0:
                best_problem = config_data.loc[config_data['convergence_rate'].idxmax()]
                print(f"  {config}: {best_problem['problem_kind']} ({best_problem['convergence_rate']:.4f})")
    
        return convergence_data, config_stats, problem_stats

    return (plot_convergence_by_preconditioner_config,)


@app.cell
def _(Filter, List, SuiteSparseDownloader):
    async def add_problem_kind(df):
        """
        from ssdownload import SuiteSparseDownloader, Filter
            Usage: `await add_problem_kind(df)`
        """
        downloader = SuiteSparseDownloader()
        matrix_names: List[str] = df["matrix_name"].unique()
        for matrix_name in matrix_names:
            group, name = matrix_name.split("/", 1)
            filter = Filter(group=group, name=name)
            matrix_info = await downloader.find_matrices(filter)
            df.loc[df["matrix_name"]==matrix_name, "problem_kind"] = matrix_info[0]["kind"]
    return (add_problem_kind,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Check Data""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("diag_ssor_results.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return


@app.cell
def _(df):
    configs_list = df["preconditioner_configs"].unique().tolist()
    return (configs_list,)


@app.cell
def _(configs_list):
    configs_list
    return


@app.cell
async def _(add_problem_kind, df):
    await add_problem_kind(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 収束性およびicholの成功率""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Converged, NOT_Converged, Failed の割合""")
    return


@app.cell
def _(configs_list, df, pd):
    # ichol の成功率の変化

    results_is_converged = []

    # filtered_configs_list = [config for config in configs_list if config != "type=none"]
    # for config in filtered_configs_list:
    for config in configs_list:
        sum = df[df["preconditioner_configs"] == config].shape[0]
        converged_count = df[(df["is_converged"] == 1) & (df["preconditioner_configs"] == config)].shape[0]
        not_converged_count = df[(df["is_converged"] == 0) & (df["preconditioner_configs"] == config)].shape[0]
        failed_ichol_count = df[(df["is_converged"] == -1) & (df["preconditioner_configs"] == config)].shape[0]

        converged_prob = converged_count / sum
        not_converged_prob = not_converged_count / sum
        failed_ichol_prob = failed_ichol_count / sum

        results_is_converged.append({
            "Config": config,
            "Converged": converged_prob,
            "NOT Converged": not_converged_prob,
            "FAILED at ichol": failed_ichol_prob
        })

        print(f"{config}:{sum}")
        print(f"CONVERGED: {converged_count} \t ({converged_prob})")
        print(f"NOT CONVERGED: {not_converged_count} \t ({not_converged_prob})")
        print(f"FAILED at building prec: {failed_ichol_count} \t ({failed_ichol_prob})")
        print("------------------------------------------------------------------------------------------------------------------------")

    results_is_converged = pd.DataFrame(results_is_converged)
    output_filename = "convergence_results.xlsx"
    results_is_converged.to_excel(output_filename, index=False)
    # results_is_converged
    return (results_is_converged,)


@app.cell
def _(pd, plt, results_is_converged):
    # データをロング形式へ変換
    long = (results_is_converged
            .melt(id_vars='Config',
                  value_vars=['Converged', 'NOT Converged', 'FAILED at ichol'],
                  var_name='Outcome',
                  value_name='Prob'))

    # Converged が高い順に並べ替え
    config_order = (results_is_converged
                    .sort_values('Converged', ascending=False)['Config'])

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(config_order)))
    bottom = pd.Series(0, index=config_order)

    palette = {
        'Converged':      '#1b9e77',
        'NOT Converged':  '#d95f02',
        'FAILED at ichol':'#7570b3',
    }

    for outcome in ['Converged', 'NOT Converged', 'FAILED at ichol']:
        vals = results_is_converged.set_index('Config').loc[config_order, outcome]
        ax.barh(config_order, vals, left=bottom,
                color=palette[outcome], label=outcome, edgecolor='white')
        bottom += vals

    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1)
    ax.set_title("Convergence Rate and ichol Factorization Failure Rate for Each Option")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 反復回数と計算時間, および前処理行列構築時間""")
    return


@app.cell
def _(df, plt, sns):
    converged_df = df[df["is_converged"] == 1]

    # iter_final/matrix_size の比率を計算
    converged_df['iter_ratio'] = converged_df['iter_final'] / converged_df['matrix_size']

    config_stats = converged_df.groupby('preconditioner_configs')['iter_ratio'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(6)

    # print("Summary statistics of iter_final per matrix_size for each preconditioner configuration:")
    # print(config_stats)

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=converged_df, x='preconditioner_configs', y='iter_ratio')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Reference (y=1.0)')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of iter_final per matrix_size, grouped by preconditioner configuration')
    plt.ylabel('iter_final / matrix_size')
    plt.tight_layout()
    plt.show()

    # base = alt.Chart(converged_df).mark_boxplot().encode(
    #     x=alt.X(
    #         'preconditioner_configs:N',
    #         axis=alt.Axis(
    #             labelAngle=-45,         # 斜めにして省スペース
    #             labelLimit=0,           # ← 切り捨て禁止
    #             title='Preconditioner Configuration'
    #         )
    #     ),
    #     y=alt.Y(
    #         'iter_ratio:Q',
    #         axis=alt.Axis(title='iter_final / matrix_size')
    #     )
    # ).properties(
    #     width=alt.Step(55),            # ← 1 カテゴリ 55 px
    #     height=400,
    #     title='Distribution of iter_final per matrix_size, grouped by preconditioner configuration'
    # )

    # reference_line = alt.Chart(
    #     pd.DataFrame({'y': [1.0]})
    # ).mark_rule(
    #     color='gray', strokeDash=[5, 5], opacity=0.7, strokeWidth=2
    # ).encode(y='y:Q')

    # mo.ui.altair_chart(base + reference_line)
    return (converged_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **簡単・解きやすい問題しか解いてないやつがはやくみえるため注意！！**

    ---

    **おおよその指標として:**

    - 0:50000 のサイズの行列で (サイズ $\times$ 0.25) ぐらいが, 目安っぽい
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### 計算時間および前処理行列構築時間""")
    return


@app.cell
def _(converged_df, plt, sns):
    # preconditioner_configs ごとの solve_time の統計量
    solve_time_stats = converged_df.groupby('preconditioner_configs')['solve_time'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(6)

    # print("preconditioner_configs ごとの solve_time の統計:")
    # print(solve_time_stats)

    # 箱ひげ図で可視化
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=converged_df, x='preconditioner_configs', y='solve_time')
    plt.xticks(rotation=45, ha='right')
    plt.title('Solve Time for Each Option')
    plt.ylabel('solve_time (s)')
    plt.yscale('log')  # 対数スケールで見やすくする
    plt.tight_layout()
    plt.show()

    # 各設定での詳細な分析
    # print("\n詳細な分析:")
    # for _config in converged_df['preconditioner_configs'].unique():
    #     subset = converged_df[converged_df['preconditioner_configs'] == _config]
    #     print(f"\n{_config}:")
    #     print(f"  ケース数: {len(subset)}")
    #     print(f"  平均 solve_time: {subset['solve_time'].mean():.6f} 秒")
    #     print(f"  中央値: {subset['solve_time'].median():.6f} 秒")
    #     print(f"  標準偏差: {subset['solve_time'].std():.6f}")
    #     print(f"  範囲: {subset['solve_time'].min():.6f} - {subset['solve_time'].max():.6f} 秒")
    return


@app.cell
def _(converged_df, plt, sns):
    # 前処理行列の構築が不要な設定を除外するリスト
    exclude_configs = [
        'type=none',
    ]

    # 除外する設定以外のデータをフィルタリング
    filtered_converged_df = converged_df[~converged_df['preconditioner_configs'].isin(exclude_configs)]

    # preconditioner_configs ごとの construction_time の統計量
    construction_time_stats = filtered_converged_df.groupby('preconditioner_configs')['construction_time'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(6)

    # print("preconditioner_configs ごとの construction_time の統計:")
    # print(construction_time_stats)

    # 箱ひげ図で可視化
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=filtered_converged_df, x='preconditioner_configs', y='construction_time')
    plt.xticks(rotation=45, ha='right')
    plt.title('Construction Time for Each Option')
    plt.ylabel('construction_time (s)')
    plt.yscale('log')  # 対数スケールで見やすくする
    plt.tight_layout()
    plt.show()

    # 各設定での詳細な分析
    # print("\n詳細な分析:")
    # for config in filtered_converged_df['preconditioner_configs'].unique():
    #     subset = filtered_converged_df[filtered_converged_df['preconditioner_configs'] == config]
    #     print(f"\n{config}:")
    #     print(f"  ケース数: {len(subset)}")
    #     print(f"  平均 construction_time: {subset['construction_time'].mean():.6f} 秒")
    #     print(f"  中央値: {subset['construction_time'].median():.6f} 秒")
    #     print(f"  標準偏差: {subset['construction_time'].std():.6f}")
    #     print(f"  範囲: {subset['construction_time'].min():.6f} - {subset['construction_time'].max():.6f} 秒")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- construction_time は diag. (ssor は前処理行列構築もくそもない)""")
    return


@app.cell
def _(df, plot_convergence_by_preconditioner_config):
    # convergence_data, config_stats, problem_stats = plot_convergence_by_preconditioner_config(df)
    _, _, _ = plot_convergence_by_preconditioner_config(df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
