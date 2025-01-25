#!/usr/bin/env python3
"""
step4_visualize_statistics_csv.py

Modifications / Revisions:
1. Reads from CSV instead of JSON.
2. Treats each (model_name, prompt_type) pair as a unique model row 
   in the plots/tables, using the naming convention:
      f"{model_name}_({prompt_type})"
3. Defines exactly two ensembles: MODEL_SUBSET_LS, MODEL_ALL_LS
4. Outputs go to evaluation_reports_summary/plots.

Enhancements:
- Make the "Top n Models"/"Bottom n Models" floating text 3x larger, bold Arial, alpha=0.3.
- Move the performance plot horizontal line labels ("XGBoost" and "Tabular LLMs") to 60% of the width.
- Use MODEL_DESCRIPTION_DT + prompt_type for the x-axis labels in create_performance_plot().
- For all table output types, replace model names with short form
  (MODEL_DESCRIPTION_DT + line break + (prompt_type)).
- For all table output types, replace metric headers with METRICS_DESCRIPTION_DT values.
- Modify the text report to list stats for **all** models in MODEL_ALL_LS (no top/bottom).
- Add generate_json(), which outputs a *.json file with each model as a key and its metrics as a dictionary.
- For all three *.png plots, if FLAG_NO_ZEROS=True, remove rows that have zero values for the relevant metrics
  *before* computing top/bottom sets so we avoid zero-value outliers.

Notes:
- Ensure that `fpdf` is installed for PDF generation: `pip install fpdf`.
- The dictionaries MODEL_DESCRIPTION_DT and METRICS_DESCRIPTION_DT should be updated if you add or remove models/metrics.

"""

import logging
import sys
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# For PDF output (ensure you have installed the fpdf package: pip install fpdf)
from fpdf import FPDF

# --------------------------------------------------------------------------
# Data Structures
# --------------------------------------------------------------------------
MODEL_DESCRIPTION_DT = {
    "athene-v2:72b-q4_K_M": "Athene-v2 (72b)",
    "aya-expanse:8b-q4_K_M": "Aya Expanse (8b,)",
    "command-r:35b-08-2024-q4_K_M": "Command-R (35b)",
    "dolphin3:8b-llama3.1-q4_K_M": "Dolphin3-Llama3.1 (8b)",
    "exaone3.5:7.8b-instruct-q4_K_M": "ExaOne3.5 (7.8b)",
    "falcon3:7b-instruct-q4_K_M": "Falcon3 (7b)",
    "gemma2:9b-instruct-q4_K_M": "Gemma2 (9b)",
    "glm4:9b-chat-q4_K_M": "GLM4 (9b)",
    "granite3.1-dense:8b-instruct-q4_K_M": "Granite3.1 Dense (8b)",
    "hermes3:8b-llama3.1-q4_K_M": "Hermes3 (8b)",
    "llama3.1:70b-instruct-q4_K_M": "Llama3.1 (70b)",
    "llama3.2:1b-instruct-q4_K_M": "Llama3.2 (1b)",
    "llama3.2:3b-instruct-q4_K_M": "Llama3.2 (3b)",
    "llama3.1:8b-instruct-q4_K_M": "Llama3.1 (8b)",
    "llama3.3:70b-instruct-q4_K_M": "Llama3.3 (70b)",
    "marco-o1:7b-q4_K_M": "Marco-o1 (7b)",
    "mistral:7b-instruct-q4_K_M": "Mistral (7b)",
    "nemotron-mini:4b-instruct-q4_K_M": "Nemotron Mini (4b)",
    "olmo2:7b-1124-instruct-q4_K_M": "Olmo2 (7b)",
    "phi4:14b-q4_K_M": "Phi4 (14b)",
    "qwen2.5:0.5b-instruct-q4_K_M": "Qwen2.5 (0.5b)",
    "qwen2.5:1.5b-instruct-q4_K_M": "Qwen2.5 (1.5b)",
    "qwen2.5:3b-instruct-q4_K_M": "Qwen2.5 (3b)",
    "qwen2.5:7b-instruct-q4_K_M": "Qwen2.5 (7b)",
    "qwen2.5:14b-instruct-q4_K_M": "Qwen2.5 (14b)",
    "qwen2.5:32b-instruct-q4_K_M": "Qwen2.5 (32b)",
    "qwen2.5:72b-instruct-q4_K_M": "Qwen2.5 (72b)",
    "smallthinker:3b-preview-q4_K_M": "Smallthinker (3b)",
    "tulu3:8b-q4_K_M": "Tulu3 (8b)",
}

METRICS_DESCRIPTION_DT = {
    "f1_score": "F1",
    "accuracy": "Acc",
    "precision": "Prec",
    "recall": "Recall",
    "execution_time_mean": "Exec (s)",
    "execution_time_sd": "Exec SD (s)",
    "eval_duration_mean": "Eval Mean (s)",
    "prompt_eval_count_mean": "Prompt Count",
    "eval_count_mean": "Eval Count",
}

# --------------------------------------------------------------------------
# Ensemble definitions
# --------------------------------------------------------------------------
MODEL_SUBSET_LS = [
    'athene-v2:72b-q4_K_M',
    'aya-expanse:8b-q4_K_M',
]

MODEL_ALL_LS = [
    "athene-v2:72b-q4_K_M",
    "aya-expanse:8b-q4_K_M",
    "command-r:35b-08-2024-q4_K_M",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "dolphin3:8b-llama3.1-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "falcon3:7b-instruct-q4_K_M",
    "gemma2:9b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "granite3.1-moe:3b-instruct-q4_K_M",
    "hermes3:8b-llama3.1-q4_K_M",
    "llama3.1:70b-instruct-q4_K_M",
    "llama3.2:1b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",  
    "marco-o1:7b-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "nemotron-mini:4b-instruct-q4_K_M",
    "olmo2:7b-1124-instruct-q4_K_M",
    "phi4:14b-q4_K_M",
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:32b-instruct-q4_K_M", 
    "qwen2.5:72b-instruct-q4_K_M", 
    "smallthinker:3b-preview-q4_K_M",
    "tulu3:8b-q4_K_M",
]

# Change this variable to either "subset" or "all" to switch ensembles
ENSEMBLE_NAME = "all"  # pick one in ['subset', 'all']

# How many models to display as top & bottom for the plots/tables
TOP_OR_BOTTOM_CT = 15

# --------------------------------------------------------------------------
# NEW Global Flag: If True, remove zero-value rows (for relevant metrics)
# before computing top/bottom sets in the plots.
# --------------------------------------------------------------------------
FLAG_NO_ZEROS = True

# --------------------------------------------------------------------------
# Configurable exclude lists (if needed)
# Leave them empty if you don't want to exclude anything
# --------------------------------------------------------------------------
EXCLUDE_MODEL_LS = []
EXCLUDED_OUTPUT_LS = []

# --------------------------------------------------------------------------
# Basic logging configuration
# --------------------------------------------------------------------------
def setup_logging():
    """Configure logging to both file and console with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --------------------------------------------------------------------------
# Utility: Parse & rename model strings and metric labels
# --------------------------------------------------------------------------
def get_pretty_model_label(model_str, use_linebreak=False):
    """
    Convert something like 'athene-v2:72b-q4_K_M_(system1)'
    into 'Athene-v2 (72b) (system1)' or 'Athene-v2 (72b)\n(system1)',
    depending on use_linebreak.
    """
    if "_(" not in model_str:
        # If there's no prompt suffix, just do a fallback
        base = model_str
        prompt_type = ""
    else:
        parts = model_str.split("_(")
        base = parts[0]
        prompt_type = parts[1].rstrip(")")  # remove trailing ')'
    
    # Look up a short label if possible
    base_label = MODEL_DESCRIPTION_DT.get(base, base)
    
    if prompt_type:
        # Decide whether to line break
        if use_linebreak:
            return f"{base_label}\n({prompt_type})"
        else:
            return f"{base_label} ({prompt_type})"
    else:
        return base_label

def get_pretty_metric_label(metric_key):
    """
    Convert metric keys like "f1_score" => "F1" 
    using METRICS_DESCRIPTION_DT.
    """
    return METRICS_DESCRIPTION_DT.get(metric_key, metric_key)

# --------------------------------------------------------------------------
# Utility / filtering
# --------------------------------------------------------------------------
def filter_and_validate_models(df, model_list):
    """
    Filters DataFrame to include only specified *base* model_names that exist 
    in the data. Because we've appended '_(prompt_type)', we check the base 
    model_name portion before suffix.
    """
    logger.info("Starting model filtering process...")

    if not model_list:
        logger.info("No model list provided. Using all available rows.")
        return df
    
    logger.info(f"Number of distinct 'model' entries in data: {len(df['model'].unique())}")
    
    # Because df['model'] is something like "athene-v2:72b-q4_K_M_(system1)",
    # let's separate out the base model from the suffix.
    df['base_model'] = df['model'].apply(lambda x: x.split('_(')[0])
    
    # Now filter to keep only those rows whose base_model is in model_list
    df_filtered = df[df['base_model'].isin(model_list)].copy()
    df_filtered.drop(columns=['base_model'], inplace=True, errors='ignore')

    logger.info(f"Filtering done. Remaining rows: {len(df_filtered)}")
    return df_filtered.reset_index(drop=True)


def remove_zero_rows_for_performance_plot(df_in):
    """
    If FLAG_NO_ZEROS=True, remove rows with f1_score == 0 or accuracy == 0
    before sorting for top/bottom. Otherwise, return df_in as-is.
    """
    if not FLAG_NO_ZEROS:
        return df_in
    return df_in[(df_in['f1_score'] != 0) & (df_in['accuracy'] != 0)]

def remove_zero_rows_for_timing_plot(df_in):
    """
    If FLAG_NO_ZEROS=True, remove rows with total_time == 0.
    """
    if not FLAG_NO_ZEROS:
        return df_in
    # We must compute total_time = execution_time_mean + eval_duration_mean
    temp = df_in.copy()
    temp['total_time'] = temp['execution_time_mean'] + temp['eval_duration_mean']
    return temp[temp['total_time'] != 0]

def remove_zero_rows_for_token_plot(df_in):
    """
    If FLAG_NO_ZEROS=True, remove rows with total_count == 0.
    """
    if not FLAG_NO_ZEROS:
        return df_in
    temp = df_in.copy()
    temp['total_count'] = temp['prompt_eval_count_mean'] + temp['eval_count_mean']
    return temp[temp['total_count'] != 0]

# --------------------------------------------------------------------------
# Plot helper: vertical divider & "Top / Bottom" text
# --------------------------------------------------------------------------
def add_top_bottom_divider_text(ax, n):
    """
    Draws a vertical line between the top-n and bottom-n bars, and places
    floating text:
      "Top n Models" on the left half
      "Bottom n Models" on the right half
    with a bigger font, bold, alpha=0.3.
    """
    ax.axvline(x=n - 0.5, color='red', linestyle='--', linewidth=1.5)

    left_xpos = (n - 1) / 2
    right_xpos = n + (n / 2)

    y_min, y_max = ax.get_ylim()
    y_text = y_max * 0.75

    # 3x larger than old 10 => 30, plus alpha=0.3, bold Arial
    ax.text(
        left_xpos, y_text, f"Top {n} Models",
        ha='center', va='center',
        fontsize=30, fontweight='bold', fontname='Arial', alpha=0.3
    )
    ax.text(
        right_xpos, y_text, f"Bottom {n} Models",
        ha='center', va='center',
        fontsize=30, fontweight='bold', fontname='Arial', alpha=0.3
    )

# --------------------------------------------------------------------------
# Plot creation functions
# --------------------------------------------------------------------------
def create_performance_plot(df, output_path, root_filename):
    """Creates a bar plot for the top-(n) and bottom-(n) models' performance metrics."""
    logger.info("Creating performance plot...")

    try:
        # If we want to remove zero-rows, do so
        df_used = remove_zero_rows_for_performance_plot(df)

        # Sort by `f1_score` descending
        df_sorted = df_used.sort_values('f1_score', ascending=False).reset_index(drop=True)

        # Take the top-(n) and bottom-(n)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0).reset_index(drop=True)

        # Build data subset
        plot_data = pd.DataFrame({
            'model': combined['model'],
            'f1_score': combined['f1_score'],
            'accuracy_normalized': combined['accuracy'] / 100.0
        })

        perf_melt = plot_data.melt(
            id_vars="model",
            value_vars=["f1_score", "accuracy_normalized"],
            var_name="metric",
            value_name="value"
        )

        # For x-axis, use MODEL_DESCRIPTION_DT + (prompt_type)
        perf_melt['display_model'] = perf_melt['model'].apply(
            lambda x: get_pretty_model_label(x, use_linebreak=False)
        )
        perf_melt['display_metric'] = perf_melt['metric'].apply(get_pretty_metric_label)

        plt.figure(figsize=(12, 7))
        custom_colors = ["#1f77b4", "#ff7f0e"]
        ax = sns.barplot(
            x="display_model", 
            y="value", 
            hue="display_metric", 
            data=perf_melt, 
            palette=custom_colors
        )

        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.title(f"Performance Metrics by Model (Top & Bottom - {ENSEMBLE_NAME.upper()} ensemble)")
        plt.ylim(0, 1)

        # Make legend lines thicker
        handles, labels = ax.get_legend_handles_labels()
        for patch in handles:
            patch.set_linewidth(3)
            patch.set_edgecolor(patch.get_facecolor())

        plt.legend(title="Metric", loc="best")

        # Add vertical divider & text
        add_top_bottom_divider_text(ax, TOP_OR_BOTTOM_CT)

        # Light horizontal lines at 0.65 and 0.85
        ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)

        # Move horizontal line labels to ~60% of the width
        x_min, x_max = ax.get_xlim()
        x_label = x_min + 0.6 * (x_max - x_min)

        ax.text(x_label, 0.65, "Tabular LLMs", color='gray', ha='center', va='bottom', fontsize=9)
        ax.text(x_label, 0.85, "XGBoost", color='gray', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        output_filename = f"{root_filename}_performance_top-bottom-{TOP_OR_BOTTOM_CT}.png"
        output_fullpath = os.path.join(output_path, output_filename)
        plt.savefig(output_fullpath)
        plt.close()

        logger.info(f"Performance plot saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error creating performance plot: {str(e)}")
        plt.close()
        raise

# [ADDED] New function to plot ALL models in descending order of F1
def create_performance_plot_all(df, output_path, root_filename):
    """
    Creates a bar plot for ALL models (no top/bottom) in descending order of F1.
    """
    logger.info("Creating performance plot for ALL models...")

    try:
        df_used = remove_zero_rows_for_performance_plot(df)

        # Sort by F1 descending
        df_sorted = df_used.sort_values('f1_score', ascending=False).reset_index(drop=True)

        plot_data = pd.DataFrame({
            'model': df_sorted['model'],
            'f1_score': df_sorted['f1_score'],
            # We'll treat "accuracy" as a 0 to 100 range, so dividing by 100 to normalize
            'accuracy_normalized': df_sorted['accuracy'] / 100.0
        })

        perf_melt = plot_data.melt(
            id_vars="model",
            value_vars=["f1_score", "accuracy_normalized"],
            var_name="metric",
            value_name="value"
        )

        # For x-axis, use short label
        perf_melt['display_model'] = perf_melt['model'].apply(
            lambda x: get_pretty_model_label(x, use_linebreak=False)
        )
        perf_melt['display_metric'] = perf_melt['metric'].apply(get_pretty_metric_label)

        plt.figure(figsize=(12, 7))
        custom_colors = ["#1f77b4", "#ff7f0e"]
        ax = sns.barplot(
            x="display_model", 
            y="value",
            hue="display_metric",
            data=perf_melt,
            palette=custom_colors
        )

        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.title(f"Performance Metrics by Model (All - {ENSEMBLE_NAME.upper()} ensemble)")
        plt.ylim(0, 1)

        # Make legend lines thicker
        handles, labels = ax.get_legend_handles_labels()
        for patch in handles:
            patch.set_linewidth(3)
            patch.set_edgecolor(patch.get_facecolor())

        plt.legend(title="Metric", loc="best")

        # Optional: add the same horizontal lines as reference
        ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)

        # [NEW] Light dashed horizontal line at 0.50 + label
        ax.axhline(y=0.50, color='gray', linestyle='--', alpha=0.5)

        # We'll position text at ~60% across the x-axis, slightly above y=0.50
        x_min, x_max = ax.get_xlim()
        x_label_50 = x_min + 0.6 * (x_max - x_min)
        ax.text(
            x_label_50, 
            0.505,  # Slightly above the 0.50 line
            "50/50 chance", 
            color='gray', 
            alpha=0.7, 
            ha='center', 
            va='bottom', 
            fontsize=9
        )

        # Also keep the earlier references for lines at 0.65 & 0.85
        ax.text(x_label_50, 0.65, "Tabular LLMs", color='gray', ha='center', va='bottom', fontsize=9)
        ax.text(x_label_50, 0.85, "XGBoost", color='gray', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save output
        output_filename = f"{root_filename}_performance_all.png"
        output_fullpath = os.path.join(output_path, output_filename)
        plt.savefig(output_fullpath)
        plt.close()

        logger.info(f"Performance (all models) plot saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error creating performance (all) plot: {str(e)}")
        plt.close()
        raise



def create_timing_plot(df, output_path, root_filename):
    """Creates a stacked bar plot for timing metrics (top & bottom models)."""
    logger.info("Creating timing plot...")

    try:
        df_used = remove_zero_rows_for_timing_plot(df)
        # Convert ns to seconds
        df_used['execution_time_mean'] = df_used['execution_time_mean'] / 1e9
        df_used['eval_duration_mean'] = df_used['eval_duration_mean'] / 1e9
        df_used['total_time'] = df_used['execution_time_mean'] + df_used['eval_duration_mean']
        df_sorted = df_used.sort_values('total_time', ascending=False).reset_index(drop=True)

        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        if FLAG_NO_ZEROS:
            non_zero_df = df_sorted[df_sorted['total_time'] > 0]
            bottom_models = non_zero_df.tail(TOP_OR_BOTTOM_CT)
        else:
            bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        
        combined = pd.concat([top_models, bottom_models], axis=0).reset_index(drop=True)
        display_labels = [get_pretty_model_label(m, use_linebreak=False) for m in combined['model']]

        plt.figure(figsize=(12, 7))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        bars1 = plt.bar(
            display_labels,
            combined['eval_duration_mean'],
            label=get_pretty_metric_label("eval_duration_mean")
        )
        bars2 = plt.bar(
            display_labels,
            combined['execution_time_mean'],
            bottom=combined['eval_duration_mean'],
            label=get_pretty_metric_label("execution_time_mean")
        )

        # Add value labels only on top
        for i, _ in enumerate(display_labels):
            total = combined.iloc[i]['total_time']
            plt.text(i, total + (total * 0.05), f'{total:.2g}',
                    ha='center', va='bottom', 
                    fontsize=int(plt.rcParams['font.size']*(.75)),
                    rotation=90, color='black')

        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.title(f"Compute (Timing) Metrics by Model (Top & Bottom - {ENSEMBLE_NAME.upper()} ensemble)")
        plt.xlabel("Model")
        plt.ylabel("Time (seconds)")
        plt.legend()

        ax_curr = plt.gca()
        add_top_bottom_divider_text(ax_curr, TOP_OR_BOTTOM_CT)

        plt.tight_layout()
        output_filename = f"{root_filename}_compute-timing_top-bottom-{TOP_OR_BOTTOM_CT}.png"
        output_fullpath = os.path.join(output_path, output_filename)
        plt.savefig(output_fullpath)
        plt.close()

        logger.info(f"Timing plot saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error creating timing plot: {str(e)}")
        plt.close()
        raise


def create_token_plot(df, output_path, root_filename):
    """Creates a stacked bar plot for token metrics (top & bottom models)."""
    logger.info("Creating token (compute) plot...")

    try:
        # Possibly remove zero-rows for token
        df_used = remove_zero_rows_for_token_plot(df)
        df_used['total_count'] = df_used['prompt_eval_count_mean'] + df_used['eval_count_mean']
        df_sorted = df_used.sort_values('total_count', ascending=False).reset_index(drop=True)

        # Take the top-(n) and bottom-(n)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0).reset_index(drop=True)

        display_labels = [
            get_pretty_model_label(m, use_linebreak=False) for m in combined['model']
        ]

        plt.figure(figsize=(12, 7))
        ax = plt.bar(
            display_labels,
            combined['prompt_eval_count_mean'],
            label=get_pretty_metric_label("prompt_eval_count_mean")
        )
        plt.bar(
            display_labels,
            combined['eval_count_mean'],
            bottom=combined['prompt_eval_count_mean'],
            label=get_pretty_metric_label("eval_count_mean")
        )

        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.title(f"Compute (Token) Metrics by Model (Top & Bottom - {ENSEMBLE_NAME.upper()} ensemble)")
        plt.xlabel("Model")
        plt.ylabel("Count")
        plt.legend()

        ax_curr = plt.gca()
        add_top_bottom_divider_text(ax_curr, TOP_OR_BOTTOM_CT)

        plt.tight_layout()

        output_filename = f"{root_filename}_compute-tokens_top-bottom-{TOP_OR_BOTTOM_CT}.png"
        output_fullpath = os.path.join(output_path, output_filename)
        plt.savefig(output_fullpath)
        plt.close()

        logger.info(f"Token plot saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error creating token plot: {str(e)}")
        plt.close()
        raise


# --------------------------------------------------------------------------
# Markdown table generator
# --------------------------------------------------------------------------
def generate_markdown_table(df, output_path, root_filename):
    """
    Generates a markdown table for the top-(n) and bottom-(n) models by f1_score,
    displaying model with line break for prompt type, and metric headers
    from METRICS_DESCRIPTION_DT.
    """
    logger.info("Generating markdown table...")

    try:
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0)

        # Prepare lines for table
        md_lines = []
        md_lines.append("# Model Performance and Compute Summary (Top & Bottom Models)")
        md_lines.append("")

        # List of metrics we display:
        metric_cols = [
            "f1_score", "accuracy", "precision", "recall", 
            "execution_time_mean", "execution_time_sd", 
            "eval_duration_mean", "prompt_eval_count_mean", "eval_count_mean"
        ]
        # Build header row
        header_str = "| Model | " + " | ".join(
            get_pretty_metric_label(col) for col in metric_cols
        ) + " |"
        md_lines.append(header_str)

        # Alignment row
        align_str = "|-------|" + "|".join(["------:"] * len(metric_cols)) + "|"
        md_lines.append(align_str)

        # Rows
        for _, row in combined.iterrows():
            model_display = get_pretty_model_label(row['model'], use_linebreak=True)
            row_str = f"| {model_display} "
            for col in metric_cols:
                val = row[col]
                row_str += f"| {val:.4f} " if "count" not in col else f"| {val:.2f} "
            row_str += "|"
            md_lines.append(row_str)

        output_filename = f"{root_filename}_table_top-bottom-{TOP_OR_BOTTOM_CT}.md"
        output_fullpath = os.path.join(output_path, output_filename)
        with open(output_fullpath, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown summary table saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error generating markdown table: {str(e)}")
        raise

# --------------------------------------------------------------------------
# HTML table generator
# --------------------------------------------------------------------------
def generate_html_table(df, output_path, root_filename):
    """
    Generates an HTML file containing the same top-(n) and bottom-(n) model data,
    with short model names and short metric labels.
    """
    logger.info("Generating HTML table...")

    try:
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0)

        metric_cols = [
            "f1_score", "accuracy", "precision", "recall", 
            "execution_time_mean", "execution_time_sd",
            "eval_duration_mean", "prompt_eval_count_mean", "eval_count_mean"
        ]
        headers = ["Model"] + [get_pretty_metric_label(m) for m in metric_cols]

        html_lines = []
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append("<meta charset='utf-8'>")
        html_lines.append("<title>Model Performance and Compute Summary (Top & Bottom)</title>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append("<h1>Model Performance and Compute Summary (Top & Bottom Models)</h1>")
        html_lines.append("<table border='1' style='border-collapse: collapse;'>")

        # Header row
        html_lines.append("<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")

        # Rows
        for _, row in combined.iterrows():
            model_display = get_pretty_model_label(row['model'], use_linebreak=False)
            row_cells = [model_display]
            for col in metric_cols:
                val = row[col]
                if "count" in col:
                    row_cells.append(f"{val:.2f}")
                else:
                    row_cells.append(f"{val:.4f}")
            html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in row_cells) + "</tr>")

        html_lines.append("</table>")
        html_lines.append("</body></html>")

        output_filename = f"{root_filename}_table_top-bottom-{TOP_OR_BOTTOM_CT}.html"
        output_fullpath = os.path.join(output_path, output_filename)
        with open(output_fullpath, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_lines))

        logger.info(f"HTML summary table saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error generating HTML table: {str(e)}")
        raise

# --------------------------------------------------------------------------
# PDF table generator
# --------------------------------------------------------------------------
def generate_pdf_table(df, output_path, root_filename):
    """
    Generates a PDF file containing the top-(n) and bottom-(n) model data,
    with short model names and short metric labels.
    """
    logger.info("Generating PDF table...")

    try:
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0)

        metric_cols = [
            "f1_score", "accuracy", "precision", "recall",
            "execution_time_mean", "execution_time_sd",
            "eval_duration_mean", "prompt_eval_count_mean", "eval_count_mean"
        ]
        headers = ["Model"] + [get_pretty_metric_label(m) for m in metric_cols]
        col_widths = [45, 13, 13, 13, 13, 16, 16, 17, 18, 18]  # Rough column widths

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Model Performance and Compute Summary (Top & Bottom Models)", ln=True, align='C')
        pdf.ln(5)

        pdf.set_font("helvetica", 'B', 14)
        # Print header
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, text=h, border=1, align='C')
        pdf.ln(8)

        pdf.set_font("Arial", '', 9)
        for _, row in combined.iterrows():
            model_display = get_pretty_model_label(row['model'], use_linebreak=False)
            row_data = [model_display]

            for col in metric_cols:
                val = row[col]
                if "count" in col:
                    row_data.append(f"{val:.2f}")
                else:
                    row_data.append(f"{val:.4f}")

            for datum, w in zip(row_data, col_widths):
                pdf.cell(w, 8, txt=datum, border=1, align='C')
            pdf.ln(8)

        output_filename = f"{root_filename}_table_top-bottom-{TOP_OR_BOTTOM_CT}.pdf"
        output_fullpath = os.path.join(output_path, output_filename)
        pdf.output(output_fullpath)

        logger.info(f"PDF summary table saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error generating PDF table: {str(e)}")
        raise

# --------------------------------------------------------------------------
# JSON generator
# --------------------------------------------------------------------------
def generate_json(df, output_path, root_filename):
    """
    Outputs a JSON file with { model_name: { metric_name: metric_value, ...}, ... }
    covering the same top-(n) and bottom-(n) sets by f1_score.
    """
    logger.info("Generating JSON output...")

    try:
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0)

        # Build a dictionary of model->metric_dict
        out_data = {}
        for _, row in combined.iterrows():
            model_key = row['model']
            out_data[model_key] = {
                "f1_score": float(row["f1_score"]),
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "execution_time_mean": float(row["execution_time_mean"]),
                "execution_time_sd": float(row["execution_time_sd"]),
                "eval_duration_mean": float(row["eval_duration_mean"]),
                "prompt_eval_count_mean": float(row["prompt_eval_count_mean"]),
                "eval_count_mean": float(row["eval_count_mean"]),
            }

        output_filename = f"{root_filename}_top-bottom-{TOP_OR_BOTTOM_CT}.json"
        output_fullpath = os.path.join(output_path, output_filename)
        with open(output_fullpath, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)

        logger.info(f"JSON summary saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error generating JSON file: {str(e)}")
        raise

# --------------------------------------------------------------------------
# Text report generator
# --------------------------------------------------------------------------
def generate_text_report(df, output_path, root_filename):
    """
    Outputs a plain-text summary listing statistics for ALL models in MODEL_ALL_LS
    (no top/bottom). This includes additional metadata, if available.
    """
    logger.info("Generating text report...")

    try:
        # Filter df to ALL base models in MODEL_ALL_LS (ignore subset or top/bottom)
        df_textreport = df.copy()
        # Re-filter for base model in MODEL_ALL_LS:
        df_textreport['base_model'] = df_textreport['model'].apply(lambda x: x.split('_(')[0])
        df_textreport = df_textreport[df_textreport['base_model'].isin(MODEL_ALL_LS)].copy()
        df_textreport.drop(columns=['base_model'], inplace=True, errors='ignore')

        # Sort by f1_score descending just so the order is consistent
        df_textreport.sort_values('f1_score', ascending=False, inplace=True)

        lines = []
        lines.append("========== TEXT REPORT SUMMARY (ALL MODELS) ==========\n")
        lines.append("Listing stats for ALL models in MODEL_ALL_LS:\n")

        for _, row in df_textreport.iterrows():
            model_display = get_pretty_model_label(row['model'], use_linebreak=False)
            lines.append(f"Model: {model_display}")
            lines.append(f"  {get_pretty_metric_label('f1_score')}: {row['f1_score']:.4f}")
            lines.append(f"  {get_pretty_metric_label('accuracy')}: {row['accuracy']:.4f}")
            lines.append(f"  {get_pretty_metric_label('precision')}: {row['precision']:.4f}")
            lines.append(f"  {get_pretty_metric_label('recall')}: {row['recall']:.4f}")
            lines.append(f"  {get_pretty_metric_label('execution_time_mean')}: {row['execution_time_mean']:.4f}")
            lines.append(f"  {get_pretty_metric_label('execution_time_sd')}: {row['execution_time_sd']:.4f}")
            lines.append(f"  {get_pretty_metric_label('eval_duration_mean')}: {row['eval_duration_mean']:.4f}")
            lines.append(f"  {get_pretty_metric_label('prompt_eval_count_mean')}: {row['prompt_eval_count_mean']:.2f}")
            lines.append(f"  {get_pretty_metric_label('eval_count_mean')}: {row['eval_count_mean']:.2f}")
            lines.append("")

        # Additional metadata if columns exist
        lines.append("Additional Metadata:\n")
        if 'api_calls' in df_textreport.columns:
            total_api_calls = df_textreport['api_calls'].sum()
            lines.append(f"  Total API Calls: {int(total_api_calls)}")

        if 'failures' in df_textreport.columns:
            total_failures = df_textreport['failures'].sum()
            lines.append(f"  Total Failures: {int(total_failures)}")
            if 'api_calls' in df_textreport.columns and total_api_calls > 0:
                success_rate = (total_api_calls - total_failures) / total_api_calls * 100
                lines.append(f"  Success Rate: {success_rate:.2f}%")

        # Example: average F1 across entire set
        avg_f1 = df_textreport['f1_score'].mean()
        lines.append(f"\n  Overall Average {get_pretty_metric_label('f1_score')}: {avg_f1:.4f}")

        lines.append("\n========== END OF TEXT REPORT ==========\n")

        output_filename = f"{root_filename}_ALL_summary_report.txt"
        output_fullpath = os.path.join(output_path, output_filename)
        with open(output_fullpath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        logger.info(f"Text summary report saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error generating text report: {str(e)}")
        raise

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    logger.info("Starting CSV-based visualization process.")

    # 1) Determine which ensemble to use
    if ENSEMBLE_NAME == "subset":
        ENSEMBLE_MODEL_LS = MODEL_SUBSET_LS
    else:
        ENSEMBLE_MODEL_LS = MODEL_ALL_LS
    
    logger.info(f"Using ENSEMBLE_NAME = '{ENSEMBLE_NAME}' with {len(ENSEMBLE_MODEL_LS)} base models.")

    # 2) Path configuration
    INPUT_FILEPATH = os.path.join("evaluation_reports_summary", "summary_all_all-standardllm.csv")
    OUTPUT_ROOT_DIR = os.path.join("evaluation_reports_summary", "plots")

    # Create output directory if needed
    if not os.path.exists(OUTPUT_ROOT_DIR):
        logger.warning(f"Output directory does not exist. Creating: {OUTPUT_ROOT_DIR}")
        os.makedirs(OUTPUT_ROOT_DIR)

    # 3) Read the CSV
    logger.info(f"Reading CSV from: {INPUT_FILEPATH}")
    if not os.path.isfile(INPUT_FILEPATH):
        logger.error(f"Cannot find input file: {INPUT_FILEPATH}")
        sys.exit(1)

    df_raw = pd.read_csv(INPUT_FILEPATH)
    logger.info(f"Successfully read {len(df_raw)} rows from CSV.")

    # 4) Combine model_name + prompt_type into new 'model'
    df_raw['model'] = df_raw['model_name'].astype(str) + "_(" + df_raw['prompt_type'].astype(str) + ")"

    # 5) Map CSV columns
    df_raw.rename(
        columns={
            'prediction_accuracy': 'accuracy',  # interpret as percentage
        },
        inplace=True
    )

    # 6) Build the final DataFrame with the columns needed
    df = df_raw[[
        'model',
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'execution_time_mean',
        'execution_time_sd',
        'eval_duration_mean',
        'prompt_eval_count_mean',
        'eval_count_mean'
    ]].copy()

    # If you have extra columns for text report stats (api_calls, failures), copy them too
    for col in ['api_calls', 'failures']:
        if col in df_raw.columns:
            df[col] = df_raw[col]

    # 7) Filter to only the base model names we want (keeping all prompt_type variants)
    df_filtered = filter_and_validate_models(df, ENSEMBLE_MODEL_LS)

    # 8) Plotting and table generation
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ROOT_FILENAME = f"stats_visualization_{ENSEMBLE_NAME}_{datetime_str}"
    
    sns.set_theme(style="whitegrid")

    # Top/Bottom tables & plots
    generate_markdown_table(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    generate_html_table(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    generate_pdf_table(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    generate_json(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)

    create_performance_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    create_timing_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    create_token_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)

    # [ADDED] Create a new plot with ALL models in descending order of F1
    create_performance_plot_all(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)

    # Text report includes ALL models in MODEL_ALL_LS (no top/bottom)
    generate_text_report(df, OUTPUT_ROOT_DIR, ROOT_FILENAME)

    logger.info("All visualization tasks completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)
