#!/usr/bin/env python3
"""
step4_visualize_statistics_csv.py

Modifications:
1. Reads from CSV instead of JSON.
2. Treats each (model_name, prompt_type) pair as a unique model row 
   in the plots/tables, using the naming convention:
      f"{model_name}_({prompt_type})"
3. Defines exactly two ensembles: MODEL_SUBSET_LS, MODEL_ALL_LS
4. Outputs go to evaluation_reports_summary/plots.

Additional Revisions (per user instructions):
- For each of the 3 bar plots, draw a vertical line to separate top-n vs bottom-n groups
  and add floating text labeling 'Top {n} Models' on the left half, 'Bottom {n} Models' on
  the right half (both at ~25% down from top of the plot).
- In the performance plot legend, increase thickness and match bar colors (no longer black lines).
- In the performance plot, add light dashed horizontal lines at y=0.65 labeled "Tabular LLMs"
  and y=0.85 labeled "XGBoost", with the labels centered at ~40% of the plot width from the left.
"""

import logging
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

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
    "dolphin3:8b-llama3.1-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "falcon3:7b-instruct-q4_K_M",
    "gemma2:9b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
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
TOP_OR_BOTTOM_CT = 10  # how many models to plot for each of the top and bottom categories

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

# --------------------------------------------------------------------------
# Plot helper: vertical divider + "Top n / Bottom n" text
# --------------------------------------------------------------------------
def add_top_bottom_divider_text(ax, n):
    """
    Draws a vertical line between the top-n and bottom-n bars, and places
    two floating text annotations: "Top n Models" on the left half, 
    "Bottom n Models" on the right half, ~25% down from the top.
    """
    # The x-values for the bars are integer-based from 0..(2n - 1)
    # We add a vertical line at x = n - 0.5 so it neatly separates them.
    ax.axvline(x=n - 0.5, color='red', linestyle='--', linewidth=1.5)

    # We'll place the text at the horizontal center of each half
    # left half center ~ n/2 - 0.5, right half center ~ n + n/2
    left_xpos = (n - 1) / 2
    right_xpos = n + (n / 2)

    # ~25% down from top => 75% of y-range
    y_min, y_max = ax.get_ylim()
    y_text = y_max * 0.75

    ax.text(left_xpos, y_text, f"Top {n} Models",
            ha='center', va='center', fontsize=10, color='black')
    ax.text(right_xpos, y_text, f"Bottom {n} Models",
            ha='center', va='center', fontsize=10, color='black')

# --------------------------------------------------------------------------
# Plot creation functions
# --------------------------------------------------------------------------
def create_performance_plot(df, output_path, root_filename):
    """Creates a bar plot for the top-(n) and bottom-(n) models' performance metrics."""
    logger.info("Creating performance plot...")

    try:
        # Sort by `f1_score` descending
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)

        # Take the top-(n) and bottom-(n)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0).reset_index(drop=True)

        # Build a data subset for plotting
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

        plt.figure(figsize=(12, 7))
        # Use a custom palette for the two metrics
        custom_colors = ["#1f77b4", "#ff7f0e"]  # e.g., default matplotlib colors
        ax = sns.barplot(
            x="model", 
            y="value", 
            hue="metric", 
            data=perf_melt, 
            palette=custom_colors
        )

        # Rotate x-ticks
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.title(f"Performance Metrics by Model (Top & Bottom Models - {ENSEMBLE_NAME.upper()} ensemble)")
        plt.ylim(0, 1)

        # ---- Make legend lines thicker & match bar colors ----
        handles, labels = ax.get_legend_handles_labels()
        for patch in handles:
            # Each patch is a rectangle for the legend entry
            patch.set_linewidth(3)
            # Ensure the edge color is the same as face color (so it isn't black)
            patch.set_edgecolor(patch.get_facecolor())

        plt.legend(title="Metric", loc="best")

        # ---- Add the vertical divider & text labeling top-n and bottom-n halves ----
        add_top_bottom_divider_text(ax, TOP_OR_BOTTOM_CT)

        # ---- Add light horizontal lines at y=0.65 and y=0.85 plus labels ----
        ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)

        # Place the textual labels near ~40% width
        x_min, x_max = ax.get_xlim()
        x_label = x_min + 0.4 * (x_max - x_min)

        ax.text(x_label, 0.65, "Tabular LLMs", color='gray',
                ha='center', va='bottom', fontsize=9)
        ax.text(x_label, 0.85, "XGBoost", color='gray',
                ha='center', va='bottom', fontsize=9)

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


def create_timing_plot(df, output_path, root_filename):
    """Creates a stacked bar plot for timing metrics (top & bottom models)."""
    logger.info("Creating timing plot...")

    try:
        df['total_time'] = df['execution_time_mean'] + df['eval_duration_mean']
        df_sorted = df.sort_values('total_time', ascending=False).reset_index(drop=True)

        # Take the top-(n) and bottom-(n)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0).reset_index(drop=True)

        plt.figure(figsize=(12, 7))
        ax = plt.bar(
            combined['model'],
            combined['eval_duration_mean'],
            label='Evaluation Duration'
        )
        ax2 = plt.bar(
            combined['model'],
            combined['execution_time_mean'],
            bottom=combined['eval_duration_mean'],
            label='Execution Time'
        )

        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.title(f"Compute (Timing) Metrics by Model (Top & Bottom Models - {ENSEMBLE_NAME.upper()} ensemble)")
        plt.xlabel("Model")
        plt.ylabel("Time (seconds)")
        plt.legend()

        # Add the vertical divider & text labeling top-n and bottom-n halves
        # We need to get the current axis as a variable, so let's do that:
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
        df['total_count'] = df['prompt_eval_count_mean'] + df['eval_count_mean']
        df_sorted = df.sort_values('total_count', ascending=False).reset_index(drop=True)

        # Take the top-(n) and bottom-(n)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0).reset_index(drop=True)

        plt.figure(figsize=(12, 7))
        ax = plt.bar(
            combined['model'],
            combined['prompt_eval_count_mean'],
            label='Prompt Eval Count'
        )
        ax2 = plt.bar(
            combined['model'],
            combined['eval_count_mean'],
            bottom=combined['prompt_eval_count_mean'],
            label='Evaluation Count'
        )

        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.title(f"Compute (Token) Metrics by Model (Top & Bottom Models - {ENSEMBLE_NAME.upper()} ensemble)")
        plt.xlabel("Model")
        plt.ylabel("Count")
        plt.legend()

        # Add the vertical divider & text labeling top-n and bottom-n halves
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
    """Generates a markdown table for the top-(n) and bottom-(n) models by f1_score."""
    logger.info("Generating markdown table...")

    try:
        # Sort by `f1_score` descending
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)

        # Take the top-(n) and bottom-(n)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0)

        md_lines = []
        md_lines.append("# Model Performance and Compute Summary (Top & Bottom Models)")
        md_lines.append("")
        md_lines.append("| model | f1_score | accuracy | precision | recall | execution_time_mean | execution_time_sd | eval_duration_mean | prompt_eval_count_mean | eval_count_mean |")
        md_lines.append("|-------|---------:|---------:|----------:|-------:|--------------------:|-------------------:|--------------------:|-----------------------:|-----------------:|")

        for _, row in combined.iterrows():
            md_lines.append(
                f"| {row['model']} "
                f"| {row['f1_score']:.4f} "
                f"| {row['accuracy']:.4f} "
                f"| {row['precision']:.4f} "
                f"| {row['recall']:.4f} "
                f"| {row['execution_time_mean']:.4f} "
                f"| {row['execution_time_sd']:.4f} "
                f"| {row['eval_duration_mean']:.4f} "
                f"| {row['prompt_eval_count_mean']:.2f} "
                f"| {row['eval_count_mean']:.2f} |"
            )

        output_filename = f"{root_filename}_table_top-bottom-{TOP_OR_BOTTOM_CT}.md"
        output_fullpath = os.path.join(output_path, output_filename)
        with open(output_fullpath, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown summary table saved to {output_fullpath}")
    except Exception as e:
        logger.error(f"Error generating markdown table: {str(e)}")
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

    # 5) Map CSV columns to standardized columns used in plotting
    #    We'll treat prediction_accuracy as "accuracy", which we expect in percentage (0-100).
    df_raw.rename(
        columns={
            'prediction_accuracy': 'accuracy',  # interpret as percentage
        },
        inplace=True
    )

    # 6) Build our final DataFrame with columns the plotting functions expect
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

    # 7) Filter to only the base model names we want (keeping all prompt_type variants)
    df_filtered = filter_and_validate_models(df, ENSEMBLE_MODEL_LS)

    # 8) Plotting and table generation
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ROOT_FILENAME = f"stats_visualization_{ENSEMBLE_NAME}_{datetime_str}"
    
    sns.set_theme(style="whitegrid")

    generate_markdown_table(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    create_performance_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    create_timing_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    create_token_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)

    logger.info("All visualization tasks completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)
