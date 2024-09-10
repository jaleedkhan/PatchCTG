import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def explore_data_results(df, plots=False, name=None):
    # Create the title
    if name is None:
        name = "Dataset"
    count = len(df)
    print(f"{'-' * ((80 - len(name) - 2) // 2)} {name} {'-' * ((80 - len(name) - 2 + 1) // 2)}")
    print(f"Count: {count}\n")

    # List to hold data for creating a stats table
    stats_data = []

    if plots:
        num_columns = 3
        num_rows = int(np.ceil(len(df.columns) / num_columns))
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 5 * num_rows))
        axes = axes.flatten()

    # Iterate over columns except 'input_signals'
    for i, col in enumerate([col for col in df.columns if col != 'input_signals']):
        if col == 'ICD_B_10':  # Special handling for 'ICD_B_10' column
            all_codes = df[col].dropna().apply(lambda x: x.split(';')).explode().str.strip()
            value_counts = all_codes.value_counts(dropna=False)

            if len(value_counts) > 5:
                top_4 = value_counts.head(4)
                other_sum = value_counts.iloc[4:].sum()
                plot_data = top_4.copy()
                plot_data['Other'] = other_sum
            else:
                plot_data = value_counts

            stats_data.append([col, len(plot_data), "; ".join([f"{val}: {count}" for val, count in plot_data.items()])])

            if plots:
                sns.barplot(x=plot_data.index, y=plot_data.values, ax=axes[i], hue=plot_data.index, palette="viridis", legend=False)
                axes[i].set_xlabel(f"ICD_B_10 ({name})", fontsize=12)
        elif df[col].dtype == 'object' or len(df[col].unique()) <= 5:
            value_counts = df[col].value_counts(dropna=False)

            stats_data.append([col, len(value_counts), "; ".join([f"{val}: {count}" for val, count in value_counts.items()])])

            if plots:
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i], hue=value_counts.index, palette="viridis", legend=False)
        else:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            stats_data.append([col, 'Numeric', f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}"])

            if plots:
                sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue', bins=30)
    
    if plots:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    # Create a DataFrame for the stats data and display it
    stats_df = pd.DataFrame(stats_data, columns=["Column", "Type/Unique Values", "Details"])
    print(stats_df.to_string(index=False))
    print('\n')


def compare_data_results(*dfs, names=None):
    if names is None:
        names = [f'Dataset {i+1}' for i in range(len(dfs))]
    elif len(names) != len(dfs):
        raise ValueError("The number of names must match the number of dataframes.")

    dfs = [df.drop(columns=['input_signals'], errors='ignore').reset_index(drop=True) for df in dfs]

    columns_set = set(dfs[0].columns)
    if any(set(df.columns) != columns_set for df in dfs):
        raise ValueError("All dataframes must have the same columns.")

    num_columns = 3
    total_plots = len(dfs[0].columns) * len(dfs)
    num_rows = int(np.ceil(total_plots / num_columns))
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 5 * num_rows))
    axes = axes.flatten()

    plot_idx = 0  # Track which plot we are on

    for col in dfs[0].columns:
        if col == 'ICD_B_10':
            for df, name in zip(dfs, names):
                all_codes = df[col].dropna().apply(lambda x: x.split(';')).explode().str.strip()
                value_counts = all_codes.value_counts()

                if len(value_counts) > 5:
                    top_4 = value_counts.head(4)
                    other_sum = value_counts.iloc[4:].sum()
                    plot_data = top_4.copy()
                    plot_data['Other'] = other_sum
                else:
                    plot_data = value_counts

                sns.barplot(x=plot_data.index, y=plot_data.values, ax=axes[plot_idx], hue=plot_data.index, palette="viridis", legend=False)
                axes[plot_idx].set_xlabel(f"ICD_B_10 ({name})", fontsize=12)
                plot_idx += 1
        else:
            for df, name in zip(dfs, names):
                if df[col].nunique() > 1:
                    sns.kdeplot(df[col], label=name, fill=True, ax=axes[plot_idx], warn_singular=False)
            axes[plot_idx].legend(title="Dataset")
            plot_idx += 1

    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    for name, df in zip(names, dfs):
        explore_data_results(df, plots=False, name=name)
