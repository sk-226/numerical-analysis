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
def _(mo):
    mo.md(r"""## Function""")
    return


@app.cell
def _():
    def create_config_labels(config_list):
        """
        Config名を "Config N" 形式にラベリングする関数
        'type=none' は 'None' として特別扱い
        """
        unique_configs = sorted(config_list)
        config_to_label = {}
        label_to_config = {}
        config_counter = 1
    
        for config in unique_configs:
            # Check specifically for 'type=none' string
            if config == 'type=none':
                config_to_label[config] = 'None'
                label_to_config['None'] = config
            else:
                label = f'Config {config_counter}'
                config_to_label[config] = label
                label_to_config[label] = config
                config_counter += 1
    
        return config_to_label, label_to_config

    def print_config_mapping_table(label_to_config):
        """設定ラベルと実際の設定名のマッピングテーブルを出力"""
        print("\n" + "="*80)
        print("PRECONDITIONER CONFIGURATION MAPPING TABLE")
        print("="*80)
        print(f"{'Label':<12} | {'Configuration'}")
        print("-" * 80)

        # Sort labels: None first, then Config 1, Config 2, ...
        sorted_labels = sorted(label_to_config.keys(), key=lambda x: (x != 'None', x))

        for label in sorted_labels:
            config = label_to_config[label]
            # Wrap long configurations for better readability
            if len(config) > 65:
                wrapped_config = config[:65] + "..."
            else:
                wrapped_config = config
            print(f"{label:<12} | {wrapped_config}")

        print("="*80)
    return create_config_labels, print_config_mapping_table


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
        convergence_data = convergence_data[convergence_data['sample_count'] >= 10]

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


@app.cell
def _(np, plt):
    def create_condition_number_groups(df):
        """条件数に基づいて3つのグループに分割"""
        df = df.copy()
    
        # 条件数グループの定義
        conditions = [
            (df['condition_number'] >= 1.0) & (df['condition_number'] < 1.0e+5),
            (df['condition_number'] >= 1.0e+5) & (df['condition_number'] < 1.0e+10), 
            (df['condition_number'] >= 1.0e+10)
        ]
    
        choices = ['low_cond', 'mid_cond', 'high_cond']
    
        df['condition_group'] = np.select(conditions, choices, default='unknown')
    
        return df

    def plot_convergence_by_condition_number(df):
        """条件数グループ別の収束性を詳細分析・可視化（改善版）"""
    
        # 条件数グループを作成
        df_grouped = create_condition_number_groups(df)
        df_grouped = df_grouped[df_grouped['condition_group'] != 'unknown']
    
        # グループの順序を定義
        condition_order = ['low_cond', 'mid_cond', 'high_cond']
        color_map = {'low_cond': '#1f77b4', 'mid_cond': '#ff7f0e', 'high_cond': '#2ca02c'}
    
        print(f"Condition Number Group Distribution:")
        group_counts = df_grouped['condition_group'].value_counts()
        for group in condition_order:  # 順序通りに表示
            if group in group_counts:
                print(f"{group}: {group_counts[group]}")
        print()
    
        # 4つのサブプロットを作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
        # 1. 収束率の比較（順序を制御）
        convergence_by_group = df_grouped.groupby(['condition_group', 'preconditioner_configs']).agg({
            'is_converged': lambda x: (x == 1).mean(),
            'matrix_name': 'count'
        }).round(4)
        convergence_by_group.columns = ['convergence_rate', 'sample_count']
        convergence_by_group = convergence_by_group.reset_index()
    
        # 収束率をピボットして順序を制御
        convergence_pivot = convergence_by_group.pivot(
            index='preconditioner_configs', 
            columns='condition_group', 
            values='convergence_rate'
        ).fillna(0)
    
        # 列の順序を指定
        convergence_pivot = convergence_pivot.reindex(columns=condition_order, fill_value=0)
    
        convergence_pivot.plot(kind='bar', ax=ax1, width=0.8, color=[color_map[col] for col in convergence_pivot.columns])
        ax1.set_title('Convergence Rate by Condition Group', fontweight='bold')
        ax1.set_ylabel('Convergence Rate')
        ax1.legend(title='Condition Group')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    
        # 2. 反復回数の分布比較（ボックスプロットに変更）
        converged_only = df_grouped[df_grouped['is_converged'] == 1]
    
        iteration_data = []
        labels = []
        for group in condition_order:
            group_data = converged_only[converged_only['condition_group'] == group]
            if len(group_data) > 0:
                iteration_data.append(group_data['iter_final'])
                labels.append(f'{group}\n(n={len(group_data)})')
            else:
                iteration_data.append([])
                labels.append(f'{group}\n(n=0)')
    
        bp = ax2.boxplot(iteration_data, labels=labels, patch_artist=True)
    
        # ボックスプロットの色を設定
        for patch, group in zip(bp['boxes'], condition_order):
            patch.set_facecolor(color_map[group])
            patch.set_alpha(0.7)
    
        ax2.set_yscale('log')  # 対数スケールで見やすく
        ax2.set_title('Iteration Count by Condition Group', fontweight='bold')
        ax2.set_ylabel('Iteration Count (log scale)')
        ax2.grid(True, alpha=0.3)
    
        # 3. 条件数 vs 反復回数の散布図（順序を保持）
        for group in condition_order:
            group_data = converged_only[converged_only['condition_group'] == group]
            if len(group_data) > 0:
                ax3.scatter(group_data['condition_number'], group_data['iter_final'], 
                           c=color_map[group], label=group, alpha=0.6, s=20)
    
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_title('Condition Number vs Iteration Count', fontweight='bold')
        ax3.set_xlabel('Condition Number (log scale)')
        ax3.set_ylabel('Iteration Count (log scale)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
        # 4. 計算時間の比較（順序を保持）
        solve_time_data = []
        time_labels = []
        for group in condition_order:
            group_data = converged_only[converged_only['condition_group'] == group]
            if len(group_data) > 0:
                solve_time_data.append(group_data['solve_time'])
                time_labels.append(f'{group}\n(n={len(group_data)})')
            else:
                solve_time_data.append([])
                time_labels.append(f'{group}\n(n=0)')
    
        bp2 = ax4.boxplot(solve_time_data, labels=time_labels, patch_artist=True)
    
        # ボックスプロットの色を設定
        for patch, group in zip(bp2['boxes'], condition_order):
            patch.set_facecolor(color_map[group])
            patch.set_alpha(0.7)
    
        ax4.set_yscale('log')
        ax4.set_title('Solve Time by Condition Group', fontweight='bold')
        ax4.set_ylabel('Solve Time (s, log scale)')
        ax4.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        # 統計サマリーを順序通りに出力
        print("=== Convergence Statistics by Condition Group ===")
        group_stats = df_grouped.groupby('condition_group').agg({
            'is_converged': [lambda x: (x == 1).mean(), lambda x: (x == 0).mean(), lambda x: (x == -1).mean()],
            'matrix_name': 'count'
        }).round(4)
        group_stats.columns = ['converged_rate', 'not_converged_rate', 'failed_rate', 'total_count']
        # 順序を制御して表示
        group_stats = group_stats.reindex(condition_order)
        print(group_stats)
    
        print("\n=== Iteration Statistics for Converged Cases ===")
        iter_stats = converged_only.groupby('condition_group')['iter_final'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(2)
        # 順序を制御して表示
        iter_stats = iter_stats.reindex(condition_order)
        print(iter_stats)
    
        return df_grouped, convergence_by_group
    return create_condition_number_groups, plot_convergence_by_condition_number


@app.cell
def _(
    create_condition_number_groups,
    create_config_labels,
    plt,
    print_config_mapping_table,
):
    def plot_convergence_by_condition_number_normalized(df):
        """条件数グループ別の収束性を詳細分析・可視化（正規化反復回数版）"""
    
        # 最小サンプル数
        min_sample_count = 10
    
        # Check for correct column name
        if 'preconditioner_configs' in df.columns:
            config_col = 'preconditioner_configs'
        elif 'preconditioner_config' in df.columns:
            config_col = 'preconditioner_config'
        else:
            print("Warning: preconditioner_config column not found")
            return None, None

        # 条件数グループを作成
        df_grouped = create_condition_number_groups(df)
        df_grouped = df_grouped[df_grouped['condition_group'] != 'unknown']

        # 正規化された反復回数を計算（収束したケースのみ）
        converged_only = df_grouped[df_grouped['is_converged'] == 1].copy()
        converged_only['iter_ratio'] = converged_only['iter_final'] / converged_only['matrix_size']

        # グループの順序を定義
        condition_order = ['low_cond', 'mid_cond', 'high_cond']
        color_map = {'low_cond': '#1f77b4', 'mid_cond': '#ff7f0e', 'high_cond': '#2ca02c'}

        print(f"Data Summary (Normalized Iteration Analysis):")
        print(f"- Dataset shape after condition grouping: {df_grouped.shape}")
        print(f"- Converged cases for analysis: {len(converged_only)}")
        group_counts = df_grouped['condition_group'].value_counts()
        for group in condition_order:  # 順序通りに表示
            if group in group_counts:
                converged_count = len(converged_only[converged_only['condition_group'] == group])
                print(f"- {group}: {group_counts[group]} total samples ({converged_count} converged)")
        print()

        # Config labeling処理
        unique_configs = df_grouped[config_col].unique()
        config_to_label, label_to_config = create_config_labels(unique_configs)

        # 4つのサブプロットを作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 収束率の比較（Config labelingを適用）- 既存と同じ
        convergence_by_group = df_grouped.groupby(['condition_group', config_col]).agg({
            'is_converged': [lambda x: (x == 1).mean(), 'count', lambda x: (x == 0).mean(), lambda x: (x == -1).mean()]
        }).round(4)
        convergence_by_group.columns = ['convergence_rate', 'sample_count', 'max_iter_rate', 'ichol_failure_rate']
        convergence_by_group = convergence_by_group.reset_index()

        # 最小サンプル数でフィルタリング
        convergence_by_group = convergence_by_group[convergence_by_group['sample_count'] >= min_sample_count]
    
        print(f"Valid combinations (>={min_sample_count} samples): {len(convergence_by_group)}")

        # Config labelを追加
        convergence_by_group['config_label'] = convergence_by_group[config_col].map(config_to_label)

        # 収束率をピボットして順序を制御
        convergence_pivot = convergence_by_group.pivot(
            index='config_label', 
            columns='condition_group', 
            values='convergence_rate'
        ).fillna(0)

        # 列と行の順序を指定
        convergence_pivot = convergence_pivot.reindex(columns=condition_order, fill_value=0)
    
        # Config labelの順序を制御 (None first, then Config 1, Config 2, ...)
        config_labels = sorted(convergence_pivot.index, key=lambda x: (x != 'None', x))
        convergence_pivot = convergence_pivot.reindex(config_labels)

        convergence_pivot.plot(kind='bar', ax=ax1, width=0.8, color=[color_map[col] for col in convergence_pivot.columns])
        ax1.set_title('Convergence Rate by Condition Group', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Convergence Rate (is_converged == 1)', fontsize=12)
        ax1.set_xlabel('Preconditioner Configuration (see mapping table below)', fontsize=12)
        ax1.legend(title='Condition Group')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. 正規化反復回数の分布比較（ボックスプロット）
        iteration_data = []
        labels = []
        for group in condition_order:
            group_data = converged_only[converged_only['condition_group'] == group]
            if len(group_data) > 0:
                iteration_data.append(group_data['iter_ratio'])
                labels.append(f'{group}\n(n={len(group_data)})')
            else:
                iteration_data.append([])
                labels.append(f'{group}\n(n=0)')

        bp = ax2.boxplot(iteration_data, labels=labels, patch_artist=True)

        # ボックスプロットの色を設定
        for patch, group in zip(bp['boxes'], condition_order):
            patch.set_facecolor(color_map[group])
            patch.set_alpha(0.7)

        ax2.set_yscale('log')  # 対数スケールで見やすく
        ax2.set_title('Normalized Iteration Count by Condition Group', fontweight='bold', fontsize=14)
        ax2.set_ylabel('iter_final / matrix_size (log scale)', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
        # 参考線 (y=1.0) を追加
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Reference (y=1.0)')

        # 3. 条件数 vs 正規化反復回数の散布図
        for group in condition_order:
            group_data = converged_only[converged_only['condition_group'] == group]
            if len(group_data) > 0:
                ax3.scatter(group_data['condition_number'], group_data['iter_ratio'], 
                           c=color_map[group], label=group, alpha=0.6, s=20)

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_title('Condition Number vs Normalized Iteration Count', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Condition Number (log scale)', fontsize=12)
        ax3.set_ylabel('iter_final / matrix_size (log scale)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
        # 参考線 (y=1.0) を追加
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

        # 4. 計算時間の比較（既存と同じ）
        solve_time_data = []
        time_labels = []
        for group in condition_order:
            group_data = converged_only[converged_only['condition_group'] == group]
            if len(group_data) > 0:
                solve_time_data.append(group_data['solve_time'])
                time_labels.append(f'{group}\n(n={len(group_data)})')
            else:
                solve_time_data.append([])
                time_labels.append(f'{group}\n(n=0)')

        bp2 = ax4.boxplot(solve_time_data, labels=time_labels, patch_artist=True)

        # ボックスプロットの色を設定
        for patch, group in zip(bp2['boxes'], condition_order):
            patch.set_facecolor(color_map[group])
            patch.set_alpha(0.7)

        ax4.set_yscale('log')
        ax4.set_title('Solve Time by Condition Group', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Solve Time (s, log scale)', fontsize=12)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(pad=3.0)
        plt.show()

        # Configuration mapping table を出力
        print_config_mapping_table(label_to_config)

        # 統計サマリーを順序通りに出力
        print("\n=== Convergence Statistics by Condition Group ===")
        group_stats = df_grouped.groupby('condition_group').agg({
            'is_converged': [lambda x: (x == 1).mean(), lambda x: (x == 0).mean(), lambda x: (x == -1).mean()],
            'matrix_name': 'count'
        }).round(4)
        group_stats.columns = ['converged_rate', 'not_converged_rate', 'failed_rate', 'total_count']
        # 順序を制御して表示
        group_stats = group_stats.reindex(condition_order)
        print(group_stats)

        print("\n=== Normalized Iteration Statistics for Converged Cases ===")
        iter_stats = converged_only.groupby('condition_group')['iter_ratio'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(4)
        # 順序を制御して表示
        iter_stats = iter_stats.reindex(condition_order)
        print(iter_stats)

        # Config別の統計
        print("\n=== Statistics by Preconditioner Configuration ===")
        config_stats = convergence_by_group.groupby(config_col).agg({
            'convergence_rate': ['mean', 'std', 'min', 'max'],
            'max_iter_rate': 'mean',
            'ichol_failure_rate': 'mean',
            'sample_count': 'sum'
        }).round(4)
        config_stats.columns = ['avg_convergence', 'std_convergence', 'min_convergence', 
                               'max_convergence', 'avg_max_iter_rate', 'avg_ichol_failure_rate', 'total_samples']
        config_stats = config_stats.reset_index().sort_values('avg_convergence', ascending=False)
        print(config_stats)

        # Notable combinations
        print("\n=== Notable Combinations ===")
    
        # Best convergence combination
        best_combo = convergence_by_group.loc[convergence_by_group['convergence_rate'].idxmax()]
        print(f"Best convergence: {best_combo['condition_group']} + {best_combo['config_label']} = {best_combo['convergence_rate']:.4f}")

        # Worst convergence combination
        worst_combo = convergence_by_group.loc[convergence_by_group['convergence_rate'].idxmin()]
        print(f"Worst convergence: {worst_combo['condition_group']} + {worst_combo['config_label']} = {worst_combo['convergence_rate']:.4f}")

        # High ichol failure rates
        high_ichol_failure = convergence_by_group[convergence_by_group['ichol_failure_rate'] > 0.1]
        if len(high_ichol_failure) > 0:
            print(f"\nCombinations with high ichol failure rates (>10%):")
            for _, row in high_ichol_failure.iterrows():
                print(f"  {row['condition_group']} + {row['config_label']}: {row['ichol_failure_rate']:.3f}")

        # Best normalized iteration performance by condition group
        print(f"\nBest normalized iteration performance by condition group:")
        for group in condition_order:
            group_data = converged_only[converged_only['condition_group'] == group]
            if len(group_data) > 0:
                best_ratio = group_data['iter_ratio'].min()
                best_matrix = group_data[group_data['iter_ratio'] == best_ratio]['matrix_name'].iloc[0]
                print(f"  {group}: {best_ratio:.4f} (matrix: {best_matrix})")

        return df_grouped, convergence_by_group
    return (plot_convergence_by_condition_number_normalized,)


@app.cell
def _(mo):
    mo.md(r"""## Check Data""")
    return


@app.cell
async def _(add_problem_kind, pd):
    df = pd.read_csv("diag_ssor_results.csv")
    cond_num_df = pd.read_csv("cond_spd_0-50000.csv")

    await add_problem_kind(df)
    df = df.merge(cond_num_df[["matrix_name", "condition_number"]], 
                  on="matrix_name", 
                  how="left")
    return (df,)


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
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""## 収束性およびicholの成功率""")
    return


@app.cell
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


@app.cell
def _(df):
    # 収束しなかった行列
    df.loc[(df["preconditioner_configs"]=="type=diag+ssor;omega=1") & (df["is_converged"]==0), "matrix_name"]
    return


@app.cell
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


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""- construction_time は diag. (ssor は前処理行列構築もくそもない)""")
    return


@app.cell
def _(df, plot_convergence_by_preconditioner_config):
    # convergence_data, config_stats, problem_stats = plot_convergence_by_preconditioner_config(df)
    _, _, _ = plot_convergence_by_preconditioner_config(df)
    return


@app.cell
def _(df, plot_convergence_by_condition_number):
    df_with_groups, convergence_analysis = plot_convergence_by_condition_number(df)
    return


@app.cell
def _(df, plot_convergence_by_condition_number_normalized):
    _, _ = plot_convergence_by_condition_number_normalized(df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
