import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, auc
from IPython.display import clear_output
import os
import shutil
import re
from datetime import datetime, timedelta
import seaborn as sns

def explore_data_results(df, flag=None, label=None, plots=False):

    # Filter dataframe
    if label is not None:
        df = df[df['label'] == label]

    # Create the title
    title = ("Complete dataset" if flag is None and label is None 
             else f"Complete dataset with label {label}" if flag is None and label is not None
             else "Train set" if flag == 'train' and label is None 
             else "Test set" if flag == 'test' and label is None 
             else f"Train subset with label {label}" if flag == 'train' and label is not None
             else f"Test subset with label {label}" if flag == 'test' and label is not None
             else "")
    
    print(f"{'-' * ((80 - len(title) - 2) // 2)} {title} {'-' * ((80 - len(title) - 2 + 1) // 2)}")

    count = len(df)
    print(f"Count: {count}\n")
    
    # List to hold data for creating a stats table
    stats_data = []
    
    if plots:
        # Set up the plot grid if plots are to be shown
        fig, axes = plt.subplots(5, 4, figsize=(20, 16))
        axes = axes.flatten()
    
    # Iterate over columns except 'input_signals'
    for i, col in enumerate([col for col in df.columns if col != 'input_signals']):
        if df[col].dtype == 'object' or len(df[col].unique()) <= 10:
            # For categorical or low cardinality columns
            unique_values = df[col].unique()
            value_counts = df[col].value_counts(dropna=False)
            if len(unique_values) <= 10:
                category_details = []
                for val in unique_values:
                    if pd.isna(val):
                        category_details.append(f"NaN: {value_counts[np.nan]}")
                    else:
                        category_details.append(f"{val}: {value_counts[val]}")
                stats_data.append([col, len(unique_values), "; ".join(category_details)])
                
                if plots:
                    # Plotting bar chart for categorical columns
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i], hue=value_counts.index, palette="viridis", legend=False)
                    axes[i].set_title(f"{col}")

            else:
                stats_data.append([col, len(unique_values), ""])
        else:
            # For numeric columns
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            stats_data.append([col, 'Numeric', f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}"])
            
            if plots:
                # Plotting histogram for numeric columns
                sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue', bins=30)
                axes[i].set_title(f"{col}")
    
    if plots:
        # Adjust layout and remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    
        plt.tight_layout()
        plt.show()
    
    # Create a DataFrame for the stats data and display it
    stats_df = pd.DataFrame(stats_data, columns=["Column", "Type/Unique Values", "Details"])
    print(stats_df.to_string(index=False))
    print('\n')

def compare_data_results(*dfs, names=None):
    # Default names for datasets if not provided
    if names is None:
        names = [f'Dataset {i+1}' for i in range(len(dfs))]
    elif len(names) != len(dfs):
        raise ValueError("The number of names must match the number of dataframes.")

    # Remove the 'input_signals', 'ICD_B_10', 'StudyID_B' columns
    dfs = [df.drop(columns=['input_signals', 'ICD_B_10', 'StudyID_B'], errors='ignore').reset_index(drop=True) for df in dfs]

    # Check if all dataframes have the same columns
    columns_set = set(dfs[0].columns)
    if any(set(df.columns) != columns_set for df in dfs):
        raise ValueError("All dataframes must have the same columns.")

    # Number of columns in the grid
    num_columns = 3
    num_rows = 7
    total_plots = num_columns * num_rows

    # Create subplots with an adjusted figure size
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 30))
    axes = axes.flatten()
    
    # Iterate over columns to compare their distributions
    for i, col in enumerate(dfs[0].columns):
        if i >= total_plots:
            break  # Avoid plotting more than the grid allows

        # Combine all dataframes for categorical or low cardinality columns
        if dfs[0][col].dtype in ['object', 'category'] or len(dfs[0][col].unique()) <= 10:
            try:
                combined_df = pd.concat(
                    [df.assign(Dataset=name) for df, name in zip(dfs, names)],
                    ignore_index=True
                )
                sns.countplot(x=col, data=combined_df, hue='Dataset', palette="viridis", ax=axes[i])
                axes[i].set_title(f"{col} (Categorical)")
            except TypeError as e:
                print(f"Skipping column {col}: {e}")
        else:
            # Plot KDEs for numeric columns
            for df, name in zip(dfs, names):
                sns.kdeplot(df[col], label=name, fill=True, ax=axes[i])
            axes[i].set_title(f"{col} (Numeric)")
            axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()