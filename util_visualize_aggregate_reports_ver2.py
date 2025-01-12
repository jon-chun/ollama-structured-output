#!/usr/bin/env python3

"""
util_visualize_aggregate_metrics.py

Generates multiple performance visualizations (Accuracy vs. F1, 
AUC-ROC vs. Execution Time, Confusion Matrix Heatmap, Grouped Bars, etc.)
based on the aggregated CSV file containing metrics from different models.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------
# 1. Global Config and Constants
# --------------------------------------------------------------------

INPUT_AGGREGATE_CSV = os.path.join(
    'aggregate_reports', 'evaluation_results_long_20250110',
    'aggregate_model_reports.csv'
)

OUTPUT_VIS_DIR = os.path.join(
    'aggregate_reports', 'evaluation_results_long_20250110',
    'plots_enhanced'
)
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)


# --------------------------------------------------------------------
# 2. Data Loading & Preparation
# --------------------------------------------------------------------

def load_data():
    """
    Load the aggregated CSV into a Pandas DataFrame.
    Convert model_params_b and any other relevant columns to numeric.
    Create or parse a 'model_family' column for high-level grouping.
    """
    df = pd.read_csv(INPUT_AGGREGATE_CSV)

    # Convert model_params_b (e.g. '7', '32') to numeric
    df['model_params_b'] = pd.to_numeric(df['model_params_b'], errors='coerce')

    # Optional: parse model_name to extract a more general 'model_family'
    df['model_family'] = df['model_name'].apply(extract_family_name)

    return df


def extract_family_name(model_name: str) -> str:
    """
    A naive example function to identify model families from model_name.
    E.g. 'llama3.2:1b-instruct-fp16' -> 'llama'
         'falcon-40b' -> 'falcon'
         'qwen2-7b'   -> 'qwen'
         'mistral'    -> 'mistral'
         'dolphin'    -> 'dolphin'
    Customize as needed for your naming patterns.
    """
    name_lower = model_name.lower()
    # Add logic or patterns for each known family
    if 'llama' in name_lower:
        return 'llama'
    elif 'falcon' in name_lower:
        return 'falcon'
    elif 'qwen' in name_lower:
        return 'qwen'
    elif 'mistral' in name_lower:
        return 'mistral'
    elif 'dolphin' in name_lower:
        return 'dolphin'
    else:
        return 'other'


# --------------------------------------------------------------------
# 3. Visualization Functions
# --------------------------------------------------------------------

def scatter_accuracy_vs_f1(df):
    """
    Scatter plot:
      - X-axis = f1_score
      - Y-axis = prediction_accuracy (%)
      - Color = auc_roc
      - Bubble size = model_params_b
    Demonstrates how multiple metrics can be combined in a single scatter,
    with optional selective annotation to avoid clutter.
    """
    # Ensure necessary columns exist
    subset_cols = ['f1_score', 'prediction_accuracy', 'auc_roc', 'model_params_b', 'model_family']
    plot_df = df.dropna(subset=subset_cols)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        x=plot_df['f1_score'],
        y=plot_df['prediction_accuracy'],
        c=plot_df['auc_roc'],
        s=plot_df['model_params_b'] * 15,  # scale bubble size by param count
        alpha=0.6,
        cmap='coolwarm'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("AUC-ROC", rotation=270, labelpad=15)

    plt.xlabel("F1 Score")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. F1 (Color=AUC-ROC, Size=Param Size)")

    # Example: annotate top-5 in accuracy or interesting outliers
    top5_acc = plot_df.nlargest(5, 'prediction_accuracy')
    for idx, row in top5_acc.iterrows():
        label_txt = f"{row['model_family']} ({row['model_params_b']}B)"
        plt.annotate(
            label_txt,
            (row['f1_score'], row['prediction_accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_VIS_DIR, "scatter_accuracy_vs_f1.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_auc_roc_vs_execution_time(df):
    """
    Plot AUC-ROC vs. Average Execution Time,
    bubble size = model_params_b, color = model_family.
    Another approach to show how param size & family affect performance vs. time.
    """
    subset_cols = ['auc_roc', 'avg_execution_time', 'model_params_b', 'model_family']
    plot_df = df.dropna(subset=subset_cols)

    plt.figure(figsize=(10, 6))
    sizes = plot_df['model_params_b'] * 15  # scale factor for bubble sizes
    unique_families = plot_df['model_family'].unique()
    color_map = dict(zip(unique_families, sns.color_palette("hsv", len(unique_families))))

    # We could do a scatter, coloring points by their model_family
    # or use seaborn's scatterplot for a built-in legend
    sc = sns.scatterplot(
        data=plot_df,
        x='avg_execution_time',
        y='auc_roc',
        size='model_params_b',  # bubble size
        hue='model_family',
        palette=color_map,
        alpha=0.7
    )

    sc.set_title("AUC-ROC vs. Execution Time (Bubble Size=Param Size, Color=Family)")
    sc.set_xlabel("Average Execution Time (s)")
    sc.set_ylabel("AUC-ROC")

    # Adjust legend:
    # Seaborn may show two legends (one for hue, one for size) 
    # so we might want to combine or reorder them as needed.
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_VIS_DIR, "auc_roc_vs_execution_time.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def facet_by_family_boxplot(df):
    """
    Faceted boxplots:
      - columns = model_family
      - x-axis = model_params_b
      - y-axis = f1_score (or accuracy, or auc_roc, etc.)
      - hue = quantization
    Helps see how param size & quantization affect performance 
    within each family (all else held constant).
    """
    subset_cols = ['f1_score', 'model_params_b', 'quantization', 'model_family']
    plot_df = df.dropna(subset=subset_cols).copy()
    plot_df['model_params_b_str'] = plot_df['model_params_b'].astype(str)

    g = sns.catplot(
        data=plot_df,
        x='model_params_b_str',
        y='f1_score',
        hue='quantization',
        col='model_family',
        kind='box',
        sharey=False,
        height=4, aspect=1.2
    )
    g.set_axis_labels("Param Size (B)", "F1 Score")
    g.set_titles("{col_name} Family")
    g.add_legend()

    plt.suptitle("F1 by Param Size & Quantization, Faceted by Family", y=1.02)
    out_path = os.path.join(OUTPUT_VIS_DIR, "facet_by_family_boxplot.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def grouped_bar_multi_metrics(df):
    """
    Show multiple performance metrics (Accuracy, F1, AUC-ROC) 
    in a grouped bar chart with facets for param size, 
    grouped by model_family, and hue=the metric itself.
    """
    metrics_of_interest = ['prediction_accuracy', 'f1_score', 'auc_roc']
    needed_cols = ['model_family', 'quantization', 'model_params_b'] + metrics_of_interest
    sub = df.dropna(subset=needed_cols).copy()

    # Melt the data
    melted = sub.melt(
        id_vars=['model_family', 'quantization', 'model_params_b'],
        value_vars=metrics_of_interest,
        var_name='metric',
        value_name='score'
    )
    # Convert param size to string if you'd like it as a categorical facet:
    melted['model_params_b_str'] = melted['model_params_b'].astype(str)

    g = sns.catplot(
        data=melted,
        x='model_family',
        y='score',
        hue='metric',
        col='model_params_b_str',
        kind='bar',
        sharey=False,
        height=4, aspect=1.3
    )
    g.set_titles("Param Size = {col_name}B")
    g.set_axis_labels("Model Family", "Score")
    g.add_legend()

    plt.suptitle("Comparison of Accuracy, F1, AUC-ROC across Families & Param Sizes", y=1.03)
    out_path = os.path.join(OUTPUT_VIS_DIR, "grouped_bar_multi_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_confusion_matrix_heatmap_aggregate(df):
    """
    Creates an aggregate 2x2 confusion matrix by summing 
    across all models (or a subset). 
    Plots it as a heatmap for quick reference.
    """
    needed_cols = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
    if not all(col in df.columns for col in needed_cols):
        print("[WARNING] Confusion matrix columns not found. Skipping plot.")
        return

    total_tp = df['true_positives'].sum()
    total_tn = df['true_negatives'].sum()
    total_fp = df['false_positives'].sum()
    total_fn = df['false_negatives'].sum()

    matrix_data = [
        [total_tn, total_fp],
        [total_fn, total_tp]
    ]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix_data,
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Pred NO', 'Pred YES'],
        yticklabels=['Actual NO', 'Actual YES']
    )
    plt.title("Aggregate Confusion Matrix Heatmap")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_VIS_DIR, "aggregate_confusion_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


# (Optional) If you had per-model confusion matrices, you could do a facet 
# approach or generate separate heatmaps for each model_name.


# --------------------------------------------------------------------
# 4. Main Execution
# --------------------------------------------------------------------

def main():
    print("[INFO] Loading aggregated CSV data...")
    df = load_data()
    print(f"[INFO] Data loaded. Shape: {df.shape}")

    if df.empty:
        print("[WARNING] No data found in CSV. Exiting.")
        return

    # Example visualizations:

    # 1) Accuracy vs. F1 (Color = AUC-ROC, Size = Param Size) with top-5 annotation
    scatter_accuracy_vs_f1(df)

    # 2) AUC-ROC vs. Execution Time (Bubble plot, color by model_family)
    plot_auc_roc_vs_execution_time(df)

    # 3) Faceted boxplots by family (x=Param Size, y=F1, hue=quantization)
    facet_by_family_boxplot(df)

    # 4) Grouped bar showing multiple metrics (Accuracy, F1, AUC-ROC)
    grouped_bar_multi_metrics(df)

    # 5) Aggregate confusion matrix heatmap (TN, FP, FN, TP)
    plot_confusion_matrix_heatmap_aggregate(df)

    print(f"[INFO] All done! Check your '{OUTPUT_VIS_DIR}' folder for PNG files.")


if __name__ == "__main__":
    main()
