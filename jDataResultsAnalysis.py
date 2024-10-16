import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import optuna
import plotly.express as px
import plotly.io as pio
import os
import plotly.graph_objects as go

def explore_optuna_results(study_name, sqlite_path):
    """
    Load an Optuna study and plot custom optimization history, parallel coordinate, 
    and hyperparameter importance plots without using Optuna's visualization module.

    Args:
    - study_name (str): The name of the Optuna study.
    - sqlite_path (str): The path to the SQLite database file.
    """
    
    # Extract the directory where the sqlite file is located
    save_dir = os.path.dirname(sqlite_path)
    
    # Load the study
    storage_url = f"sqlite:///{sqlite_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    # 1. Optimization History Plot
    trials = study.trials
    trial_numbers = [trial.number for trial in trials if trial.value is not None]
    values = [trial.value for trial in trials if trial.value is not None]
    
    plt.figure(figsize=(12, 5))
    plt.plot(trial_numbers, values, marker='o', label='Trials')
    
    # Highlight the trial with the max Validation AUC
    max_value = max(values)
    max_index = values.index(max_value)
    max_trial_number = trial_numbers[max_index]
    
    plt.scatter(max_trial_number, max_value, edgecolor='black', s=100, zorder=5)
    plt.text(max_trial_number, max_value + 0.0075, f'{max_value:.4f}', fontsize=10, verticalalignment='bottom')
    
    # Set y-axis limits with some padding
    min_val = min(values)
    max_val = max(values)
    ymin = round(min_val - 0.05, 1)  # Round down and add some space
    ymax = round(max_val + 0.1, 1)  # Round up and add some space
    plt.ylim(ymin, ymax)
    plt.xlim(0, len(trials))
    
    plt.xlabel('Trial')
    plt.ylabel('Validation AUC')
    plt.grid(True)
    
    # Save the plot
    history_plot_path = os.path.join(save_dir, 'optimization_history.png')
    plt.savefig(history_plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Optimization history plot saved to: {history_plot_path}")
    
    # 2. Parallel Coordinate Plot
    
    # Extract parameter names and prepare data
    trials = study.trials
    params = study.best_trial.params.keys()
    
    param_values = {param: [] for param in params}
    objectives = []
    
    for trial in trials:
        if trial.value is not None:
            objectives.append(trial.value)
            for param in params:
                param_values[param].append(trial.params.get(param, np.nan))
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(param_values)
    df['Validation AUC'] = objectives
    
    # Convert learning_rate, activation, distil, and embed_type to string for plotting and sort them
    df['learning_rate'] = df['learning_rate'].apply(lambda x: f'{x:.1e}')
    unique_lr_values = sorted(df['learning_rate'].unique(), key=lambda x: float(x))
    
    df['activation'] = df['activation'].astype(str)
    unique_activation_values = sorted(df['activation'].unique())
    
    # Sort the DataFrame by 'Validation AUC' in descending order
    df = df.sort_values(by='Validation AUC', ascending=False)
    
    # Determine the min and max of the Validation AUC for the color scale
    min_auc = df['Validation AUC'].min()
    max_auc = df['Validation AUC'].max()
    
    # Create dimensions for parallel coordinates
    dimensions = []
    for param in params:
        if param == 'learning_rate':
            dimensions.append(
                dict(
                    label='learning_rate', 
                    values=[unique_lr_values.index(val) for val in df['learning_rate']],
                    tickvals=list(range(len(unique_lr_values))),
                    ticktext=unique_lr_values,
                    range=[0, len(unique_lr_values) - 1]
                )
            )
        elif param == 'activation':
            dimensions.append(
                dict(
                    label='activation', 
                    values=[unique_activation_values.index(val) for val in df['activation']],
                    tickvals=list(range(len(unique_activation_values))),
                    ticktext=unique_activation_values,
                    range=[0, len(unique_activation_values) - 1]
                )
            )
        else:
            dimensions.append(dict(label=param, values=df[param]))
    
    # Add Validation AUC as the last dimension
    dimensions.append(dict(label='Validation AUC', values=df['Validation AUC']))
    
    # Create the parallel coordinates plot using Plotly
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df['Validation AUC'],
                colorscale='Tealrose',
                showscale=True,
                cmin=min_auc,
                cmax=max_auc
            ),
            dimensions=dimensions
        )
    )
    
    # Save the plot as an HTML file
    parallel_coordinates_path = os.path.join(save_dir, 'parallel_coordinates.html')
    fig.write_html(parallel_coordinates_path)
    
    print(f"Parallel coordinates plot saved to: {parallel_coordinates_path}")

    
    # # 3. Hyperparameter Importance Plot
    
    # importances = optuna.importance.get_param_importances(study)
    # sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    # param_names = [item[0] for item in sorted_importances]
    # param_importances = [item[1] for item in sorted_importances]
    
    # # Reverse the order of parameters and importances for the barh plot
    # param_names.reverse()
    # param_importances.reverse()
    
    # plt.figure(figsize=(10, 5))
    # bars = plt.barh(param_names, param_importances)
    # plt.xlabel('Importance')
    # plt.title('Hyperparameter Importances')
    # plt.grid(True)
    
    # # Add the importance values to the bars
    # for bar, importance in zip(bars, param_importances):
    #     plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{importance:.4f}', 
    #              va='center', ha='left', fontsize=10)
    
    # # Save the plot
    # importance_plot_path = os.path.join(save_dir, 'hyperparameter_importance.png')
    # plt.savefig(importance_plot_path, bbox_inches='tight')
    # plt.close()
    
    # print(f"Hyperparameter importance plot saved to: {importance_plot_path}")



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
