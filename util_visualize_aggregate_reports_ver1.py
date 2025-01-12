#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------
# 1. Global Config and Constants
# --------------------------------------------------------------------

# Path to the aggregated CSV file
INPUT_AGGREGATE_CSV = os.path.join(
    'aggregate_reports', 'evaluation_results_long_20250110', 
    'aggregate_model_reports.csv'
)

# Directory where we want to store the plots
OUTPUT_VIS_DIR = os.path.join('aggregate_reports', 'evaluation_results_long_20250110', 'plots')
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)


# --------------------------------------------------------------------
# 2. Load and Prep Data
# --------------------------------------------------------------------

def load_aggregate_data(csv_path):
    """
    Load the aggregate CSV into a Pandas DataFrame.
    Convert any string columns to numeric if needed.
    """
    df = pd.read_csv(csv_path)

    # Convert model_params_b to numeric (e.g., '7' -> 7.0)
    #   If some rows might be missing or 'unknown', coerce to NaN
    df['model_params_b'] = pd.to_numeric(df['model_params_b'], errors='coerce')

    # If your CSV includes 'txt_accuracy' or other numeric columns that might be strings, convert them:
    # df['txt_accuracy'] = pd.to_numeric(df['txt_accuracy'], errors='coerce')

    return df


# --------------------------------------------------------------------
# 3. Visualization Functions
# --------------------------------------------------------------------

def plot_accuracy_vs_modelsize_quant(df):
    """
    Plot Accuracy vs. Model Size, colored by Quantization.
    Saves as 'accuracy_vs_modelsize_quant.png'.
    """
    # Filter out any rows without numeric model_params_b
    plot_df = df.dropna(subset=['model_params_b', 'prediction_accuracy'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x='model_params_b', 
        y='prediction_accuracy',
        hue='quantization',
        style='quantization',
        s=150
    )
    plt.title("Accuracy vs. Model Size / Quantization")
    plt.xlabel("Model Parameter Size (B)")
    plt.ylabel("Prediction Accuracy (%)")
    plt.legend(title="Quantization")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_VIS_DIR, "accuracy_vs_modelsize_quant.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_auc_roc_vs_execution_time(df):
    """
    Plot AUC-ROC vs. Average Execution Time, bubble size = model_params_b
    Saves as 'auc_vs_time_bubble.png'.
    """
    plot_df = df.dropna(subset=['auc_roc', 'avg_execution_time', 'model_params_b'])

    plt.figure(figsize=(10, 6))

    # Create a scatter plot with bubble size
    #   Scale bubble size (e.g. model_params_b * some factor)
    sizes = plot_df['model_params_b'] * 20  # adjust as needed
    scatter = plt.scatter(
        x=plot_df['avg_execution_time'],
        y=plot_df['auc_roc'],
        s=sizes,
        alpha=0.6,
        c=plot_df['model_params_b'],  # color by param size
        cmap='viridis'
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Model Parameter Size (B)", rotation=270, labelpad=15)
    plt.xlabel("Average Execution Time (s)")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC vs. Execution Time (Bubble Size = Parameter Size)")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_VIS_DIR, "auc_vs_time_bubble.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_confusion_matrix_heatmap(df):
    """
    OPTIONAL – Confusion matrix heatmap. We can do this in different ways:
      1) For a single model/prompt_type
      2) Summed across rows (aggregated confusion)
      3) Facet by model / prompt_type

    Below is a simple example that aggregates everything across the entire DataFrame.
    """
    # Sum across all rows to get total TP, TN, FP, FN
    total_tp = df['true_positives'].sum()
    total_tn = df['true_negatives'].sum()
    total_fp = df['false_positives'].sum()
    total_fn = df['false_negatives'].sum()

    # Build a 2x2 matrix
    matrix_data = [
        [total_tn, total_fp],
        [total_fn, total_tp]
    ]
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred NO', 'Pred YES'],
                yticklabels=['Actual NO', 'Actual YES'])
    plt.title("Aggregate Confusion Matrix Heatmap")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_VIS_DIR, "aggregate_confusion_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_stacked_bar_actual_vs_pred(df):
    """
    OPTIONAL – If your CSV had columns for 'prediction_distribution' or 'actual_distribution'
    (like the earlier examples with {NO: 55, YES: 45}),
    you could parse them and create a side-by-side or stacked bar chart.

    Since the sample CSV doesn't show 'prediction_distribution' or 'actual_distribution' columns,
    this function is just a placeholder.
    """
    pass


def plot_time_series_execution_stats(df):
    """
    OPTIONAL – If your data had a 'start_time' or 'end_time' or some timestamp column,
    you could convert it to a datetime and plot how avg_execution_time changes over time.

    The sample CSV doesn't show time columns, so this is just a placeholder.
    """
    pass


def plot_parameter_count_vs_eval(df):
    """
    OPTIONAL – If your CSV had a column with eval_count or similar, you could do:
      X-axis = model_params_b
      Y-axis = eval_count
      color/size = average_execution_time or accuracy, etc.

    Since the sample CSV doesn't show an 'eval_count' (only total_attempts or successful_attempts),
    we could do something like total_attempts vs model_params_b.
    """
    if 'total_attempts' not in df.columns:
        return

    plot_df = df.dropna(subset=['model_params_b', 'total_attempts'])

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x='model_params_b',
        y='total_attempts',
        hue='quantization'
    )
    plt.title("Total Attempts by Model Parameter Size and Quantization")
    plt.xlabel("Model Parameter Size (B)")
    plt.ylabel("Total Attempts")
    plt.legend(title="Quantization")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_VIS_DIR, "param_count_vs_attempts.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


# --------------------------------------------------------------------
# 4. Main Script
# --------------------------------------------------------------------
def main():
    print("[INFO] Loading aggregated CSV data...")
    df = load_aggregate_data(INPUT_AGGREGATE_CSV)
    print(f"[INFO] Data loaded. Shape: {df.shape}")

    # Ensure we have some data
    if df.empty:
        print("[WARNING] The CSV is empty. Exiting.")
        return

    # 1. Accuracy vs. Model Size / Quantization
    plot_accuracy_vs_modelsize_quant(df)

    # 2. AUC-ROC vs. Execution Time (Bubble plot)
    plot_auc_roc_vs_execution_time(df)

    # 3. Confusion Matrix Heatmap (Aggregate)
    plot_confusion_matrix_heatmap(df)

    # 4. (Optional) Stacked bar: Actual vs. Predicted Distribution
    #    Currently a placeholder unless you have distribution columns
    # plot_stacked_bar_actual_vs_pred(df)

    # 5. (Optional) Time Series of Execution Statistics
    #    Currently a placeholder, depends on timestamps
    # plot_time_series_execution_stats(df)

    # 6. Parameter Count vs. Evaluation Count (Example w/ total_attempts)
    plot_parameter_count_vs_eval(df)

    print(f"[INFO] All plots saved to: {OUTPUT_VIS_DIR}")


if __name__ == "__main__":
    main()
