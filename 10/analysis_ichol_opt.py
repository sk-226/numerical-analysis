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

    warnings.filterwarnings("ignore")
    return Filter, SuiteSparseDownloader, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Functions""")
    return


@app.cell
def _(plt):
    plt.rcParams['figure.figsize'] = (24, 12)
    return


@app.cell
def _(np, plt, print_config_mapping_table, sns):
    def plot_convergence_by_preconditioner_config(df):
        """Visualize convergence rates by preconditioner configuration with line plots for each problem kind"""

        min_sample_count = 10
        # 1. Data preprocessing
        # Check for correct column name
        if 'preconditioner_configs' in df.columns:
            config_col = 'preconditioner_configs'
        elif 'preconditioner_config' in df.columns:
            config_col = 'preconditioner_config'
        else:
            print("Warning: preconditioner_config column not found")
            return None, None, None

        # Calculate convergence rates for each combination (only is_converged==1 counts as success)
        convergence_data = df.groupby(['problem_kind', config_col]).agg({
            'is_converged': [lambda x: (x == 1).mean(), 'count', lambda x: (x == 0).mean(), lambda x: (x == -1).mean()]
        }).round(4)

        convergence_data.columns = ['convergence_rate', 'sample_count', 'max_iter_rate', 'ichol_failure_rate']
        convergence_data = convergence_data.reset_index()

        # Filter out combinations with insufficient samples (minimum 3 samples)
        convergence_data = convergence_data[convergence_data['sample_count'] >= min_sample_count]

        print(f"Data Summary:")
        print(f"- Number of problem kinds: {convergence_data['problem_kind'].nunique()}")
        print(f"- Number of preconditioner configs: {convergence_data[config_col].nunique()}")
        print(f"- Valid combinations: {len(convergence_data)}")
        print(f"- Convergence rate calculation: (is_converged == 1) / total_samples")
        print(f"- is_converged values: 1=success, 0=max_iter_reached, -1=ichol_failure")

        # Create index mapping for x-axis with "Config N" format (except for type=none -> None)
        unique_configs = sorted(convergence_data[config_col].unique())
        config_to_label = {}
        config_counter = 1

        for config in unique_configs:
            # Check specifically for 'type=none' string
            if config == 'type=none':
                config_to_label[config] = 'None'
            else:
                config_to_label[config] = f'Config {config_counter}'
                config_counter += 1

        label_to_config = {label: config for config, label in config_to_label.items()}
        convergence_data['config_label'] = convergence_data[config_col].map(config_to_label)

        # Sort by config label for consistent ordering (None first, then Config 1, 2, ...)
        convergence_data = convergence_data.sort_values('config_label', 
                                                       key=lambda x: x.map(lambda y: (y != 'None', y)))

        # Set up color palette for problem kinds
        problem_kinds = convergence_data['problem_kind'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(problem_kinds)))
        color_map = dict(zip(problem_kinds, colors))

        # Define line styles and markers for visual separation
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']

        # Create separate figures for each plot
        plot_convergence_line_graph(convergence_data, config_to_label, problem_kinds, 
                                   color_map, line_styles, markers)

        plot_sample_count_heatmap(df, config_col, config_to_label)

        # Print configuration mapping table
        print_config_mapping_table(label_to_config)

        # Statistical summaries and analysis
        config_stats, problem_stats = print_analysis_results(convergence_data, config_col)

        return convergence_data, config_stats, problem_stats

    def plot_convergence_line_graph(convergence_data, config_to_label, problem_kinds, 
                                   color_map, line_styles, markers):
        """Create the convergence rate line graph with Config N labels"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # Get unique config labels and create position mapping
        unique_labels = sorted(convergence_data['config_label'].unique(), 
                              key=lambda x: (x != 'None', x))
        label_positions = {label: i for i, label in enumerate(unique_labels)}

        # Main line plot with visual separation techniques
        for i, problem_kind in enumerate(problem_kinds):
            data_subset = convergence_data[convergence_data['problem_kind'] == problem_kind]

            # Sort data for proper line connections
            data_subset = data_subset.sort_values('config_label', 
                                                key=lambda x: x.map(lambda y: (y != 'None', y)))

            # Use config label positions for x-axis
            x_values = [label_positions[label] for label in data_subset['config_label']]

            # Add small jitter to y-values to separate overlapping lines
            jitter_amount = 0.005  # Small offset
            jitter = np.random.uniform(-jitter_amount, jitter_amount, len(data_subset))
            y_values = data_subset['convergence_rate'] + jitter

            # Cycle through line styles and markers
            line_style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]

            # Vary line width slightly
            line_width = 2 + (i % 3) * 0.5

            ax.plot(x_values, y_values, 
                    linestyle=line_style, marker=marker, label=problem_kind, 
                    color=color_map[problem_kind], linewidth=line_width, 
                    markersize=6, alpha=0.85, markerfacecolor='white', 
                    markeredgewidth=1.5)

        ax.set_xlabel('Preconditioner Configuration (see mapping table below)', fontsize=12)
        ax.set_ylabel('Convergence Rate (is_converged == 1)', fontsize=12)
        ax.set_title('Convergence Rate by Preconditioner Configuration and Problem Kind', 
                    fontsize=16, fontweight='bold')
        ax.set_ylim(-0.02, 1.08)  # Extended range to show separated lines better
        ax.grid(True, alpha=0.3)

        # Set x-axis ticks and labels to show Config N format
        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels(unique_labels, rotation=45, fontsize=10)
        ax.margins(x=0.01)  # Add small margin to prevent cutting off edge points

        # Legend settings with better layout for many items
        if len(problem_kinds) > 10:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=9)
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=10)

        plt.tight_layout(pad=3.0)  # Add more padding to prevent label cutoff
        plt.show()

    def plot_sample_count_heatmap(df, config_col, config_to_label):
        """Create the sample count heatmap with Config N labels"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        # Sample count heatmap for validation with config labels
        sample_counts = df.groupby(['problem_kind', config_col]).size().reset_index(name='count')
        sample_counts['config_label'] = sample_counts[config_col].map(config_to_label)

        # Create pivot table and ensure proper ordering of columns
        sample_pivot = sample_counts.pivot(index='problem_kind', columns='config_label', values='count').fillna(0)

        # Reorder columns to put None first, then Config 1, Config 2, etc.
        cols = list(sample_pivot.columns)
        ordered_cols = sorted(cols, key=lambda x: (x != 'None', x))
        sample_pivot = sample_pivot[ordered_cols]

        sns.heatmap(sample_pivot, annot=True, fmt='g', cmap='Blues', 
                    ax=ax, cbar_kws={'label': 'Sample Count'})
        ax.set_title('Sample Count by Problem Kind and Preconditioner Configuration', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Preconditioner Configuration (see mapping table below)', fontsize=12)
        ax.set_ylabel('Problem Kind', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)

        plt.tight_layout(pad=3.0)
        plt.show()

    def print_analysis_results(convergence_data, config_col):
        """Print statistical analysis results"""
        # 3. Statistical summary
        print("\n=== Statistics by Preconditioner Configuration ===")
        config_stats = convergence_data.groupby(config_col).agg({
            'convergence_rate': ['mean', 'std', 'min', 'max'],
            'max_iter_rate': 'mean',
            'ichol_failure_rate': 'mean',
            'sample_count': 'sum'
        }).round(4)
        config_stats.columns = ['avg_convergence', 'std_convergence', 'min_convergence', 
                               'max_convergence', 'avg_max_iter_rate', 'avg_ichol_failure_rate', 'total_samples']
        config_stats = config_stats.reset_index().sort_values('avg_convergence', ascending=False)
        print(config_stats)

        print("\n=== Statistics by Problem Kind ===")
        problem_stats = convergence_data.groupby('problem_kind').agg({
            'convergence_rate': ['mean', 'std', 'min', 'max'],
            'max_iter_rate': 'mean',
            'ichol_failure_rate': 'mean',
            'sample_count': 'sum'
        }).round(4)
        problem_stats.columns = ['avg_convergence', 'std_convergence', 'min_convergence', 
                                'max_convergence', 'avg_max_iter_rate', 'avg_ichol_failure_rate', 'total_samples']
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

        # Configurations with high ichol failure rates
        high_ichol_failure = convergence_data[convergence_data['ichol_failure_rate'] > 0.1]
        if len(high_ichol_failure) > 0:
            print(f"\nConfigurations with high ichol failure rates (>10%):")
            for _, row in high_ichol_failure.iterrows():
                print(f"  {row['problem_kind']} + {row[config_col]}: {row['ichol_failure_rate']:.3f}")

        # Most effective problem kind for each preconditioner config
        print("\nMost effective problem kind for each preconditioner config:")
        for config in convergence_data[config_col].unique():
            config_data = convergence_data[convergence_data[config_col] == config]
            if len(config_data) > 0:
                best_problem = config_data.loc[config_data['convergence_rate'].idxmax()]
                print(f"  {config}: {best_problem['problem_kind']} ({best_problem['convergence_rate']:.4f})")

        return config_stats, problem_stats
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
        """条件数グループ別の収束性を詳細分析・可視化（ichol style）"""
    
        # 最小サンプル数
        min_sample_count = 10
    
        # Check for correct column name
        if 'preconditioner_configs' in df.columns:
            config_col = 'preconditioner_configs'
        elif 'preconditioner_config' in df.columns:
            config_col = 'preconditioner_config'
        else:
            print("Warning: preconditioner_config column not found")
            return None, None, None

        # 条件数グループを作成
        df_grouped = create_condition_number_groups(df)
        df_grouped = df_grouped[df_grouped['condition_group'] != 'unknown']

        # グループの順序を定義
        condition_order = ['low_cond', 'mid_cond', 'high_cond']
        color_map = {'low_cond': '#1f77b4', 'mid_cond': '#ff7f0e', 'high_cond': '#2ca02c'}

        print(f"Data Summary:")
        print(f"- Dataset shape after condition grouping: {df_grouped.shape}")
        group_counts = df_grouped['condition_group'].value_counts()
        for group in condition_order:  # 順序通りに表示
            if group in group_counts:
                print(f"- {group}: {group_counts[group]} samples")
        print()

        # Config labeling処理
        unique_configs = df_grouped[config_col].unique()
        config_to_label, label_to_config = create_config_labels(unique_configs)

        # 4つのサブプロットを作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 収束率の比較（Config labelingを適用）
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

        # 2. 反復回数の分布比較（ボックスプロット）
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
        ax2.set_title('Iteration Count by Condition Group', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Iteration Count (log scale)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 3. 条件数 vs 反復回数の散布図
        for group in condition_order:
            group_data = converged_only[converged_only['condition_group'] == group]
            if len(group_data) > 0:
                ax3.scatter(group_data['condition_number'], group_data['iter_final'], 
                           c=color_map[group], label=group, alpha=0.6, s=20)

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_title('Condition Number vs Iteration Count', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Condition Number (log scale)', fontsize=12)
        ax3.set_ylabel('Iteration Count (log scale)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 計算時間の比較
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

        print("\n=== Iteration Statistics for Converged Cases ===")
        iter_stats = converged_only.groupby('condition_group')['iter_final'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(2)
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

        return df_grouped, convergence_by_group
    return (
        create_condition_number_groups,
        create_config_labels,
        plot_convergence_by_condition_number,
        print_config_mapping_table,
    )


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Check Data""")
    return


@app.cell
async def _(add_problem_kind, pd):
    df = pd.read_csv("cmp_ichol_opt_results.csv")
    cond_num_df = pd.read_csv("cond_spd_0-50000.csv")

    await add_problem_kind(df)
    df = df.merge(cond_num_df[["matrix_name", "condition_number"]], 
                  on="matrix_name", 
                  how="left")
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
def _():
    configs = [
          "struct('type','none');",
          "struct('type','ic','ictype','nofill','droptol',0.0);",
          "struct('type','ic','ictype','nofill','droptol',0.0,'michol','on');",
          "struct('type','ic','ictype','nofill','droptol',0.0,'diagcomp','max(sum(abs(A),2)./diag(A))-2');",
          "struct('type','ic','ictype','nofill','droptol',0.0,'michol','on','diagcomp','max(sum(abs(A),2)./diag(A))-2');",
          "struct('type','ic','ictype','ict','droptol',1e-1);",
          "struct('type','ic','ictype','ict','droptol',1e-1,'michol','on');",
          "struct('type','ic','ictype','ict','droptol',1e-1,'diagcomp','max(sum(abs(A),2)./diag(A))-2');",
          "struct('type','ic','ictype','ict','droptol',1e-1,'michol','on','diagcomp','max(sum(abs(A),2)./diag(A))-2');",
          "struct('type','ic','ictype','ict','droptol',1e-2);",
          "struct('type','ic','ictype','ict','droptol',1e-2,'michol','on');",
          "struct('type','ic','ictype','ict','droptol',1e-2,'diagcomp','max(sum(abs(A),2)./diag(A))-2');",
          "struct('type','ic','ictype','ict','droptol',1e-2,'michol','on','diagcomp','max(sum(abs(A),2)./diag(A))-2');",
          "struct('type','ic','ictype','ict','droptol',1e-3);",
          "struct('type','ic','ictype','ict','droptol',1e-3,'michol','on');",
          "struct('type','ic','ictype','ict','droptol',1e-3,'diagcomp','max(sum(abs(A),2)./diag(A))-2');",
          "struct('type','ic','ictype','ict','droptol',1e-3,'michol','on','diagcomp','max(sum(abs(A),2)./diag(A))-2')"
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 前処理""")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 収束性およびicholの成功率""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Converged, NOT_Converged, Failed_ichol の割合""")
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
        print(f"FAILED at ichol: {failed_ichol_count} \t ({failed_ichol_prob})")
        print("------------------------------------------------------------------------------------------------------------------------")

    results_is_converged = pd.DataFrame(results_is_converged)
    output_filename = "convergence_results.xlsx"
    results_is_converged.to_excel(output_filename, index=False)
    # results_is_converged
    return (results_is_converged,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - diagcomp オプション (今回の場合, $\alpha=max(sum(abs(A),2)./diag(A))-2$) をつけると, ichol が失敗する割合がかなり減った. 👈
    - また, $\alpha$ の決め方について, 確実に ichol を成功させるために max をとっている.
        - $\alpha$ が小さいほうが収束性がよいことから, ichol が成功するぎりぎりで収束性をよくしたい.
        - また, 一般的にもしかしたら max ではなく, ave や median でもいい可能性もある.
        - どちらにしろ, 結局実験的に $\alpha$ の値を決める必要がある.

    ---

    - michol (修正不完全コレスキー分解) を用いると, (デフォルト設定 && michol on/off) で比較したときに, なぜか michol=on のほうが ichol を失敗している?!
        - [x] なぜ?!具体的な行列名でチェックする必要あり.

    ---

    - droptol について, そのあたいより小さい値は捨てる.
        - つまり, droptol が小さいほど, ichol 時の fill ratio が dense になり, 大きいほど sparse になる.
            - 参照: [MATLAB-ichol](https://jp.mathworks.com/help/matlab/ref/incompletecholeskyfactorizationexample_02_ja_JP.png)
        - つまり, droptol が小さいほど収束ははやくなる ($M M^{-1} \approx I$).
    """
    )
    return


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
def _(mo):
    mo.md(
        r"""
    - droptol 思ってたよりも大した事ない??
        - matlab 公式のイメージが過剰に見せている気もする?
        - 1.0e-3 以下になると, fill ratio がかなり上がる
        - 精度と計算量のトレードオフが顕著になると予想される
    - michol が逆効果...

    **以下の結果見るようにかなりケースバイケースっぽい**
    """
    )
    return


@app.cell
def _(df):
    # michol オプションによる悪い効果と良い効果の比較

    config_michol_on = "type=ic;ictype=nofill;droptol=0;michol=on"
    config_michol_off = "type=ic;ictype=nofill;droptol=0"

    filtered_df_michol_on = df[df["preconditioner_configs"] == config_michol_on]
    filtered_df_michol_off = df[df["preconditioner_configs"] == config_michol_off]

    converged_michol_on = filtered_df_michol_on[filtered_df_michol_on["is_converged"] == 1]
    failed_ichol_michol_on = filtered_df_michol_on[filtered_df_michol_on["is_converged"] == -1]
    converged_michol_off = filtered_df_michol_off[filtered_df_michol_off["is_converged"] == 1]
    failed_ichol_michol_off = filtered_df_michol_off[filtered_df_michol_off["is_converged"] == -1]

    # diff matrix_name between converged_michol_on and converged_michol_off (-> michol effect which is gotten worse)
    converged_michol_on_matrix_names = converged_michol_on["matrix_name"].unique()
    failed_ichol_michol_on_matrix_names = failed_ichol_michol_on["matrix_name"].unique()
    converged_michol_off_matrix_names = converged_michol_off["matrix_name"].unique()
    failed_ichol_michol_off_matrix_names = failed_ichol_michol_off["matrix_name"].unique()

    print("COUNT:")
    print(f"converged_michol_on_matrix_names: {converged_michol_on_matrix_names.shape[0]}")
    print(f"failed_ichol_michol_on_matrix_names: {failed_ichol_michol_on_matrix_names.shape[0]}")
    print(f"converged_michol_off_matrix_names: {converged_michol_off_matrix_names.shape[0]}")
    print(f"failed_ichol_michol_off_matrix_names: {failed_ichol_michol_off_matrix_names.shape[0]}")

    # それぞれの matrix_name の差を調べる
    # michol=off のときに ichol が成功 (収束) していたのに, michol=on で icholが失敗した行列
    diff_matrix_name_michol_bad_effect = set(converged_michol_off_matrix_names) & set(failed_ichol_michol_on_matrix_names)
    # michol=off のときに失敗していたのに, michol=on で 成功 (収束) した行列
    diff_matrix_name_michol_good_effect = set(failed_ichol_michol_off_matrix_names) & set(converged_michol_on_matrix_names)

    print("------------------------------------------------------------------------------------------------------------------------")
    print("DIFF MATRIX NAME")
    print("michol=off のときに ichol が成功 (収束) していたのに, michol=on で icholが失敗した行列")
    print("COUNT:",len(diff_matrix_name_michol_bad_effect))
    print(diff_matrix_name_michol_bad_effect)
    print("----------")
    print("michol=off のときに失敗していたのに, michol=on で 成功 (収束) した行列")
    print("COUNT:",len(diff_matrix_name_michol_good_effect))
    print(diff_matrix_name_michol_good_effect)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### diagcomp=on/off + michol=on の効果""")
    return


@app.cell
def _(df):
    # diagcomp + michol オプションによる悪い効果と良い効果の比較

    config_michol_on_diagcomp_on = "type=ic;ictype=nofill;droptol=0;michol=on;diagcomp=max(sum(abs(A),2)./diag(A))-2"
    config_michol_on_diagcomp_off = "type=ic;ictype=nofill;droptol=0;michol=on"

    filtered_df_michol_on_diagcomp_on = df[df["preconditioner_configs"] == config_michol_on_diagcomp_on]
    filtered_df_michol_on_diagcomp_off = df[df["preconditioner_configs"] == config_michol_on_diagcomp_off]

    converged_michol_on_diagcomp_on = filtered_df_michol_on_diagcomp_on[filtered_df_michol_on_diagcomp_on["is_converged"] == 1]
    not_converged_michol_on_diagcomp_on = filtered_df_michol_on_diagcomp_on[filtered_df_michol_on_diagcomp_on["is_converged"] == 0]
    failed_ichol_michol_on_diagcomp_on = filtered_df_michol_on_diagcomp_on[filtered_df_michol_on_diagcomp_on["is_converged"] == -1]
    converged_michol_on_diagcomp_off = filtered_df_michol_on_diagcomp_off[filtered_df_michol_on_diagcomp_off["is_converged"] == 1]
    not_converged_michol_on_diagcomp_off = filtered_df_michol_on_diagcomp_off[filtered_df_michol_on_diagcomp_off["is_converged"] == 0]
    failed_ichol_michol_on_diagcomp_off = filtered_df_michol_on_diagcomp_off[filtered_df_michol_on_diagcomp_off["is_converged"] == -1]

    # diff matrix_name between converged_diagcomp_on and converged_diagcomp_off (-> michol effect which is gotten worse)
    converged_michol_on_diagcomp_on_matrix_names = converged_michol_on_diagcomp_on["matrix_name"].unique()
    not_converged_michol_on_diagcomp_on_matrix_names = not_converged_michol_on_diagcomp_on["matrix_name"].unique()
    failed_ichol_michol_on_diagcomp_on_matrix_names = failed_ichol_michol_on_diagcomp_on["matrix_name"].unique()
    converged_michol_on_diagcomp_off_matrix_names = converged_michol_on_diagcomp_off["matrix_name"].unique()
    not_converged_michol_on_diagcomp_off_matrix_names = not_converged_michol_on_diagcomp_off["matrix_name"].unique()
    failed_ichol_michol_on_diagcomp_off_matrix_names = failed_ichol_michol_on_diagcomp_off["matrix_name"].unique()

    print("COUNT (michol=on):")
    print(f"converged_michol_on_diagcomp_on_matrix_names: {converged_michol_on_diagcomp_on_matrix_names.shape[0]}")
    print(f"not_converged_michol_on_diagcomp_on_matrix_names: {not_converged_michol_on_diagcomp_on_matrix_names.shape[0]}")
    print(f"failed_ichol_michol_on_diagcomp_on_matrix_names: {failed_ichol_michol_on_diagcomp_on_matrix_names.shape[0]}")
    print(f"converged_michol_on_diagcomp_off_matrix_names: {converged_michol_on_diagcomp_off_matrix_names.shape[0]}")
    print(f"not_converged_michol_on_diagcomp_off_matrix_names: {not_converged_michol_on_diagcomp_off_matrix_names.shape[0]}")
    print(f"failed_ichol_michol_on_diagcomp_off_matrix_names: {failed_ichol_michol_on_diagcomp_off_matrix_names.shape[0]}")

    # それぞれの matrix_name の差を調べる
    # diagcomp=off のときに ichol が成功 (収束) していたのに, diagcomp=on で icholが失敗した行列
    diff_matrix_name_diag_bad_effect_michol_on = set(converged_michol_on_diagcomp_off_matrix_names) & set(failed_ichol_michol_on_diagcomp_on_matrix_names)
    # diagcomp=off のときに失敗していたのに, diagcomp=on で 成功 (収束) した行列
    diff_matrix_name_diag_good_effect_michol_on = set(failed_ichol_michol_on_diagcomp_off_matrix_names) & set(converged_michol_on_diagcomp_on_matrix_names)
    diff_matrix_name_diag_good_effect_include_not_converged_michol_on = set(failed_ichol_michol_on_diagcomp_off_matrix_names) & (set(converged_michol_on_diagcomp_on_matrix_names)|set(not_converged_michol_on_diagcomp_on_matrix_names))

    print("------------------------------------------------------------------------------------------------------------------------")
    print("DIFF MATRIX NAME")
    print("diagcomp=off のときに ichol が 収束 していたのに, diagcomp=on で icholが失敗した行列")
    print("COUNT:",len(diff_matrix_name_diag_bad_effect_michol_on))
    print(diff_matrix_name_diag_bad_effect_michol_on)
    print("----------")
    print("diagcomp=off のときに失敗していたのに, diagcomp=on で 収束 した行列")
    print("COUNT:",len(diff_matrix_name_diag_good_effect_michol_on))
    print(diff_matrix_name_diag_good_effect_michol_on)
    print("----------")
    print("diagcomp=off のときに失敗していたのに, diagcomp=on で 収束または非収束 した行列 (上のゆるい条件)")
    print("COUNT:",len(diff_matrix_name_diag_good_effect_include_not_converged_michol_on))
    print(diff_matrix_name_diag_good_effect_include_not_converged_michol_on)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **100% ichol 成功できなかった！！**

    - なぜ michol=off のほうが ichol が成功し, on のときに失敗するのか??
        - HB, JGD_Trefethen のグループが多い
    - 逆に off で失敗していて, on で成功する行列について,
        - Oberwolfach がおおい?? (n が小さいが 👈)
    - なにかしらの問題の性質がありそう

    ---

    - ただし, michol=on によって, ichol が成功になる例も少ないがある...
    - この境界線は？行列のどのような性質で??
        - グループ, 問題の kind でみてみる？
    - 前処理難しい, 問題 (係数行列) による 👈

    --- 

    #### **そもそもの ichol における michol オプションの理解**

    - IC(0) を例として, スパース性および要素のパターンを保持するためにフィルインは保存しない. しかし, その捨てた分について (行全体のつり合いを崩さないよう) 同じ行の対角成分に吸収 (加算/減算) する.
    - なぜ michol をするかという点について, フィルインを捨てると本来考えていた係数行列 $A$ と性質がズレてしまう (問題の対象である行列の性質として, "各行の総和が0" がある場合, その性質がくずれる).
    - ここで, **捨てた (ズレた) 分を対角成分へ動かす**ことで, (行和) $\approx \boldsymbol{0}$ を保つ.
        - ichol が失敗するとき pivot が負になると失敗する. そのことを考えると, 対角成分について, 対角成分に吸収するときに減算が起きてしまっている??

    - diagcomp も同時に入れたら最強？
        - 僅差だが, michol なしとそんなに変わらなく成った！！ (ただ, 結局 michol があると, ない場合に比べて小さい...)
            - "type=ic;ictype=nofill;droptol=0;diagcomp=max(sum(abs(A),2)./diag(A))-2"
            - "type=ic;ictype=nofill;droptol=0;michol=on;diagcomp=max(sum(abs(A),2)./diag(A))-2"
        - droptol=0.001 + michol=on がめちゃ僅差だが一応 1位！！
            - diagcomp との併用が良さそう.
    """
    )
    return


@app.cell
def _(df):
    # FUTURE WORK のために, top2 の config (michol差) で ichol が失敗している行列名を出す
    config_none="type=none"
    config_best = "type=ic;ictype=ict;droptol=0.001;michol=on;diagcomp=max(sum(abs(A),2)./diag(A))-2"
    config_best_without_michol = "type=ic;ictype=ict;droptol=0.001;diagcomp=max(sum(abs(A),2)./diag(A))-2"

    filtered_df_none = df[df["preconditioner_configs"] == config_none]
    filtered_df_config_best = df[df["preconditioner_configs"] == config_best]
    filtered_df_config_best_without_michol = df[df["preconditioner_configs"] == config_best_without_michol]

    not_converged_df_none = filtered_df_none[filtered_df_none["is_converged"] == 0]
    failed_ichol_config_best = filtered_df_config_best[filtered_df_config_best["is_converged"] == -1]
    failed_ichol_config_best_without_michol = filtered_df_config_best_without_michol[filtered_df_config_best_without_michol["is_converged"] == -1]

    not_converged_df_none_matrix_names = not_converged_df_none["matrix_name"].unique()
    failed_ichol_config_best_matrix_names = failed_ichol_config_best["matrix_name"].unique()
    failed_ichol_config_best_without_michol_matrix_names = failed_ichol_config_best_without_michol["matrix_name"].unique()

    print("COUNT:")
    print(f"not_converged_none: {not_converged_df_none_matrix_names.shape[0]}")
    print(f"failed_ichol_config_best_matrix_names: {failed_ichol_config_best_matrix_names.shape[0]}")
    print(f"failed_ichol_config_best_without_michol_matrix_names: {failed_ichol_config_best_without_michol_matrix_names.shape[0]}")

    failed_ichol_all_cases = set(failed_ichol_config_best_without_michol_matrix_names) & set(failed_ichol_config_best_matrix_names)
    failed_ichol_none_cases = set(not_converged_df_none_matrix_names) & set(failed_ichol_config_best_without_michol_matrix_names) & set(failed_ichol_config_best_matrix_names)
    print(f"ichol のbest config および best config without michol の失敗している COUNT: {len(failed_ichol_all_cases)}")
    print(f"none も含めたときのbest および best without michol の失敗している COUNT: {len(failed_ichol_none_cases)}")

    print("------------------------------------------------------------------------------------------------------------------------")
    print(failed_ichol_all_cases)
    print(failed_ichol_none_cases)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - best な config でもできていないやつはできていない. 以下その行列の list.

    ```
    {'Pothen/bodyy6', 'Boeing/bcsstm39', 'Pothen/mesh2em5', 'GHS_psdef/jnlbrng1', 'GHS_psdef/minsurfo', 'Oberwolfach/t2dal_e', 'HB/nos6', 'Pothen/mesh3e1', 'HB/bcsstm25', 'HB/bcsstm24', 'HB/bcsstm02', 'Bates/Chem97ZtZ', 'HB/bcsstm19', 'Pothen/bodyy4', 'Pothen/mesh1em1', 'HB/bcsstm20', 'Pothen/mesh1em6', 'Norris/fv1', 'Pothen/mesh2e1', 'HB/bcsstm11', 'HB/bcsstm26', 'HB/bcsstm08', 'Pothen/mesh1e1', 'HB/bcsstm06', 'HB/bcsstm22', 'Norris/fv2', 'HB/bcsstm05', 'Oberwolfach/t3d
    ```

    - ちなみに一部しか見れてないけど bcsstm26 がdiag+ic でbest_config でうまくいってた！ (diag+ssor のほうが反復回数が小さかったけど...)

    - これらに対して, diag+ssor および diag+ic で, どのような結果になるか
        - ただの SSOR では収束率は良かったが, ichol ほど反復回数を少なくすることができなかった. diag+SSOR の場合はただのSSORおよび, 上の行列たちに対して, どの程度前処理の効果が出るのか check
        - ichol については, そもそも ichol を成功さえすればよいため, ichol 成功率 (converged || not_converged) を見る.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### diagcomp 単体での効果""")
    return


@app.cell
def _(df):
    # diagcomp オプションによる悪い効果と良い効果の比較

    config_diagcomp_on = "type=ic;ictype=nofill;droptol=0;diagcomp=max(sum(abs(A),2)./diag(A))-2"
    config_diagcomp_off = "type=ic;ictype=nofill;droptol=0"

    config_diagcomp_on = "type=ic;ictype=nofill;droptol=0;diagcomp=max(sum(abs(A),2)./diag(A))-2"
    config_diagcomp_off = "type=ic;ictype=nofill;droptol=0"

    filtered_df_diagcomp_on = df[df["preconditioner_configs"] == config_diagcomp_on]
    filtered_df_diagcomp_off = df[df["preconditioner_configs"] == config_diagcomp_off]

    converged_diagcomp_on = filtered_df_diagcomp_on[filtered_df_diagcomp_on["is_converged"] == 1]
    not_converged_diagcomp_on = filtered_df_diagcomp_on[filtered_df_diagcomp_on["is_converged"] == 0]
    failed_ichol_diagcomp_on = filtered_df_diagcomp_on[filtered_df_diagcomp_on["is_converged"] == -1]
    converged_diagcomp_off = filtered_df_diagcomp_off[filtered_df_diagcomp_off["is_converged"] == 1]
    not_converged_diagcomp_off = filtered_df_diagcomp_off[filtered_df_diagcomp_off["is_converged"] == 0]
    failed_ichol_diagcomp_off = filtered_df_diagcomp_off[filtered_df_diagcomp_off["is_converged"] == -1]

    converged_diagcomp_on_matrix_names = converged_diagcomp_on["matrix_name"].unique()
    not_converged_diagcomp_on_matrix_names = not_converged_diagcomp_on["matrix_name"].unique()
    failed_ichol_diagcomp_on_matrix_names = failed_ichol_diagcomp_on["matrix_name"].unique()
    converged_diagcomp_off_matrix_names = converged_diagcomp_off["matrix_name"].unique()
    not_converged_diagcomp_off_matrix_names = not_converged_diagcomp_off["matrix_name"].unique()
    failed_ichol_diagcomp_off_matrix_names = failed_ichol_diagcomp_off["matrix_name"].unique()

    print("COUNT:")
    print(f"converged_diagcomp_on_matrix_names: {converged_diagcomp_on_matrix_names.shape[0]}")
    print(f"not_converged_diagcomp_on_matrix_names: {not_converged_diagcomp_on_matrix_names.shape[0]}")
    print(f"failed_ichol_diagcomp_on_matrix_names: {failed_ichol_diagcomp_on_matrix_names.shape[0]}")
    print("COUNT (diagcomp=off):")
    print(f"converged_diagcomp_off_matrix_names: {converged_diagcomp_off_matrix_names.shape[0]}")
    print(f"not_converged_diagcomp_off_matrix_names: {not_converged_diagcomp_off_matrix_names.shape[0]}")
    print(f"failed_ichol_diagcomp_off_matrix_names: {failed_ichol_diagcomp_off_matrix_names.shape[0]}")

    diff_matrix_name_diag_bad_effect = set(converged_diagcomp_off_matrix_names) & set(failed_ichol_diagcomp_on_matrix_names)
    diff_matrix_name_diag_good_effect = set(failed_ichol_diagcomp_off_matrix_names) & set(converged_diagcomp_on_matrix_names)
    diff_matrix_name_diag_good_effect_include_not_converged = set(failed_ichol_diagcomp_off_matrix_names) & (set(converged_diagcomp_on_matrix_names)|set(not_converged_diagcomp_on_matrix_names))

    print("------------------------------------------------------------------------------------------------------------------------")
    print("DIFF MATRIX NAME")
    print("diagcomp=off のときに ichol が 収束 していたのに, diagcomp=on で icholが失敗した行列")
    print("COUNT:",len(diff_matrix_name_diag_bad_effect))
    print(diff_matrix_name_diag_bad_effect)
    print("----------")
    print("diagcomp=off のときに失敗していたのに, diagcomp=on で 収束 した行列")
    print("COUNT:",len(diff_matrix_name_diag_good_effect))
    print(diff_matrix_name_diag_good_effect)
    print("----------")
    print("diagcomp=off のときに失敗していたのに, diagcomp=on で 収束または非収束 した行列 (上のゆるい条件)")
    print("COUNT:",len(diff_matrix_name_diag_good_effect_include_not_converged))
    print(diff_matrix_name_diag_good_effect_include_not_converged)
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

    - diagcomp 設定をすることで, ichol が成功したが, 解くためには反復回数が必要な行列なので, droptol オプションのみのものよりも反復回数が多く必要に見える
    - diagcomp によって, ichol が成功したものを見るべき！！ -> そいつらは解くのが大変, 難しい行列なはず. (ベンチマーク的な存在になるかも！！) 👈

    ---

    **おおよその指標として:**

    - 0:50000 のサイズの行列で (サイズ $\times$ 0.25) ぐらいが, 目安っぽい
    """
    )
    return


@app.cell
def _(df):
    # diagcomp によって成功した, 解くのが難しいであろう行列の名前をだす
    config_w_diagcomp = "type=ic;ictype=nofill;droptol=0;diagcomp=max(sum(abs(A),2)./diag(A))-2"
    config_wo_diagcomp = "type=ic;ictype=nofill;droptol=0"

    filtered_df_w_diagcomp = df[df["preconditioner_configs"] == config_w_diagcomp]
    filtered_df_wo_diagcomp = df[df["preconditioner_configs"] == config_wo_diagcomp]

    converged_w_diagcomp = filtered_df_w_diagcomp[filtered_df_w_diagcomp["is_converged"] == 1]
    not_converged_w_diagcomp = filtered_df_w_diagcomp[filtered_df_w_diagcomp["is_converged"] == 0]
    failed_ichol_w_diagcomp = filtered_df_w_diagcomp[filtered_df_w_diagcomp["is_converged"] == -1]
    converged_wo_diagcomp = filtered_df_wo_diagcomp[filtered_df_wo_diagcomp["is_converged"] == 1]
    not_converged_wo_diagcomp = filtered_df_wo_diagcomp[filtered_df_wo_diagcomp["is_converged"] == 0]
    failed_ichol_wo_diagcomp = filtered_df_wo_diagcomp[filtered_df_wo_diagcomp["is_converged"] == -1]

    # diff matrix_name between converged_w_diagcomp and converged_wo_diagcomp (-> michol effect which is gotten worse)
    converged_w_diagcomp_matrix_names = converged_w_diagcomp["matrix_name"].unique()
    not_converged_w_diagcomp_matrix_names = not_converged_w_diagcomp["matrix_name"].unique()
    failed_ichol_w_diagcomp_matrix_names = failed_ichol_w_diagcomp["matrix_name"].unique()
    converged_wo_diagcomp_matrix_names = converged_wo_diagcomp["matrix_name"].unique()
    not_converged_wo_diagcomp_matrix_names = not_converged_wo_diagcomp["matrix_name"].unique()
    failed_ichol_wo_diagcomp_matrix_names = failed_ichol_wo_diagcomp["matrix_name"].unique()

    print("COUNT:")
    print(f"converged_w_diagcomp_matrix_names: {converged_w_diagcomp_matrix_names.shape[0]}")
    print(f"not_converged_w_diagcomp_matrix_names: {not_converged_w_diagcomp_matrix_names.shape[0]}")
    print(f"failed_ichol_w_diagcomp_matrix_names: {failed_ichol_w_diagcomp_matrix_names.shape[0]}")
    print(f"converged_wo_diagcomp_matrix_names: {converged_wo_diagcomp_matrix_names.shape[0]}")
    print(f"not_converged_wo_diagcomp_matrix_names: {not_converged_wo_diagcomp_matrix_names.shape[0]}")
    print(f"failed_ichol_wo_diagcomp_matrix_names: {failed_ichol_wo_diagcomp_matrix_names.shape[0]}")

    # diagcomp が効いたケース (wo_diagcomp -> w_diagcomp)
    good_effect_by_diagcomp_matrix_names = set(failed_ichol_wo_diagcomp_matrix_names) & set(converged_w_diagcomp_matrix_names)
    # diagcomp でも難しい (diagcomp がついていても not 収束, または ichol 失敗している) ケース
    difficult_matrix_names = set(not_converged_w_diagcomp_matrix_names) | set(failed_ichol_w_diagcomp_matrix_names)
    # diagcomp により, off のときは収束していた (ichol 成功していた) のに, on (max...) にしたら ichol 失敗したケース
    failed_ichol_by_diagcomp = set(converged_wo_diagcomp_matrix_names) & set(failed_ichol_w_diagcomp_matrix_names)
    # wo_diagcomp のときには収束していたのに, on (max...) にしたら収束しなくなったケース
    conv_to_not = set(converged_wo_diagcomp_matrix_names) & set(not_converged_w_diagcomp_matrix_names) 

    print("------------------------------------------------------------------------------------------------------------------------")

    print("⭕IMPROVED")
    print("diagcomp が効いたケース: ", len(good_effect_by_diagcomp_matrix_names))
    print("COUNT:")
    print(good_effect_by_diagcomp_matrix_names)

    print()
    print("------------------------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------------------------")

    print("❌WORSE:")
    print("diagcomp により, off のときは収束していた (ichol 成功していた) のに, on (max...) にしたら失敗したケース")
    print("COUNT:", len(failed_ichol_by_diagcomp))
    print(failed_ichol_by_diagcomp)


    print()
    print("------------------------------------------------------------------------------------------------------------------------")
    print("diagcomp により収束→不収束になった行列")
    print("COUNT:", len(conv_to_not))
    print(sorted(conv_to_not))

    print("")
    print("------------------------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------------------------")

    print("FUTURE WORK: diagcomp でも難しい (diagcomp がついていても not 収束, または ichol 失敗している) ケース")
    print(len(difficult_matrix_names))
    print(difficult_matrix_names)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ```
    diagcomp により収束したケース
    COUNT:
    44
    ```

    - これをみると diagcomp により逆にichol が失敗しているものもある？
        - <- len(failed_ichol_by_diagcomp)=0 だった
        - すなわち, diagcomp をつけたことによって ichol が失敗する可能性はないが,
        - **converged していたものが逆にnot_convergedになるものもある **
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
    mo.md(
        r"""
    - construction_time について, diagcomp の処理は行列サイズによってかなりの時間になる. 👈
    - droptol および michol による構築時間の変化はたいしたことない
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - ichol が成功したかどうか
    - 収束性,  (反復回数, 時間)
    - ~~(真の) 残差, 誤差の振る舞い, (前処理を用いたら収束性が良くなった場合)~~
    - diagcomp はトレードオフがあることがわかった. 収束 -> 非収束になったのは 1個だけだったが, どれくらい収束率が悪くなったのか残差を比較 (桁数 etc.)
        - そして, max_iter までいっているのか. <- 反復回すだけで収束するなら, それはそれでいいため 

    - diag+ssor, diag+ic の場合で同様の検証
        - +alpha として, ただの ichol および オプションぐりぐりどけなかった行列に対しては？

    FUTURE WORK
    - 条件数が大きいものに絞ってどうか？
        - → 高精度で試して, 比較
    """
    )
    return


@app.cell
def _(df, plot_convergence_by_preconditioner_config):
    _, _, _ = plot_convergence_by_preconditioner_config(df)
    return


@app.cell
def _(df, plot_convergence_by_condition_number):
    _, _ = plot_convergence_by_condition_number(df)
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
