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

Enhancements (per user instructions):
- Use MODEL_DESCRIPTION_DT and METRICS_DESCRIPTION_DT to simplify model name and metric labeling
  across all output files (plots, tables, text report).
- Where we see 'f"{model_name}_({prompt_type})"', display something like
  'f"{MODEL_DESCRIPTION_DT[model_name]} ({prompt_type})"' in plots, or
  'f"{MODEL_DESCRIPTION_DT[model_name]}\n({prompt_type})"' in tables, etc.

Other instructions from earlier steps remain:
- For each of the 3 bar plots, draw a vertical line to separate top-n vs bottom-n groups
  and add floating text labeling 'Top {n} Models' on the left half, 'Bottom {n} Models' on
  the right half (both at ~25% down from the top of the chart).
- In the performance plot legend, increase thickness and match bar colors (no longer black lines).
- In the performance plot, add light dashed horizontal lines at y=0.65 labeled "Tabular LLMs"
  and y=0.85 labeled "XGBoost", with the labels centered at ~40% width from the left margin.

New Functions:
- generate_html_table(): Exports the same top-bottom data to an .html file.
- generate_pdf_table(): Exports the same top-bottom data to a .pdf file (requires 'fpdf' library).
- generate_text_report(): Exports a richer plain-text summary with extra metadata and stats.
"""

import logging
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# For PDF output (ensure you have installed the fpdf package: pip install fpdf)
from fpdf import FPDF

# --------------------------------------------------------------------------
# Data Structures (Provided)
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
# Utility: Parse & rename model strings and metric labels
# --------------------------------------------------------------------------
def get_pretty_model_label(model_str, use_linebreak=False):
    """
    Convert something like:
      'athene-v2:72b-q4_K_M_(system1)'
    into:
      'Athene-v2 (72b) (system1)'   or   'Athene-v2 (72b)\n(system1)'
    depending on use_linebreak.

    We:
      1. Split at '_(' to separate the base model from the prompt type.
      2. Look up a short name from MODEL_DESCRIPTION_DT for the base model.
      3. Append the prompt type in parentheses. 
    """
    if "_(" not in model_str:
        # If there's no prompt suffix, just do a fallback
        base = model_str
        prompt_type = ""
    else:
        parts = model_str.split("_(")
        base = parts[0]
        prompt_type = parts[1].rstrip(")")  # remove trailing )
    
    # Look up a short label if possible
    base_label = MODEL_DESCRIPTION_DT.get(base, base)
    
    # If there's a prompt type, we add it in parentheses
    if prompt_type:
        # Decide whether to insert line break
        if use_linebreak:
            return f"{base_label}\n({prompt_type})"
        else:
            return f"{base_label} ({prompt_type})"
    else:
        return base_label

def get_pretty_metric_label(metric_key):
    """
    Convert metric keys like "f1_score" => "F1", 
    "execution_time_mean" => "Exec (s)", etc.,
    via METRICS_DESCRIPTION_DT. Fallback if not found.
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

        # Melt so we can bar-plot by "metric"
        perf_melt = plot_data.melt(
            id_vars="model",
            value_vars=["f1_score", "accuracy_normalized"],
            var_name="metric",
            value_name="value"
        )

        # Create display columns with simpler names
        perf_melt['display_model'] = perf_melt['model'].apply(
            lambda x: get_pretty_model_label(x, use_linebreak=False)  # No line break on x-axis
        )
        perf_melt['display_metric'] = perf_melt['metric'].apply(get_pretty_metric_label)

        plt.figure(figsize=(12, 7))
        # Use a custom palette for the two metrics
        custom_colors = ["#1f77b4", "#ff7f0e"]
        ax = sns.barplot(
            x="display_model", 
            y="value", 
            hue="display_metric", 
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
            patch.set_linewidth(3)
            # Ensure the edge color is the same as face color
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

        # For simpler x-axis labels:
        display_labels = [get_pretty_model_label(m, use_linebreak=False) for m in combined['model']]

        plt.figure(figsize=(12, 7))
        ax = plt.bar(
            display_labels,
            combined['eval_duration_mean'],
            label=get_pretty_metric_label("eval_duration_mean")
        )
        plt.bar(
            display_labels,
            combined['execution_time_mean'],
            bottom=combined['eval_duration_mean'],
            label=get_pretty_metric_label("execution_time_mean")
        )

        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.title(f"Compute (Timing) Metrics by Model (Top & Bottom Models - {ENSEMBLE_NAME.upper()} ensemble)")
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
        df['total_count'] = df['prompt_eval_count_mean'] + df['eval_count_mean']
        df_sorted = df.sort_values('total_count', ascending=False).reset_index(drop=True)

        # Take the top-(n) and bottom-(n)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0).reset_index(drop=True)

        # For simpler x-axis labels:
        display_labels = [get_pretty_model_label(m, use_linebreak=False) for m in combined['model']]

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
        plt.title(f"Compute (Token) Metrics by Model (Top & Bottom Models - {ENSEMBLE_NAME.upper()} ensemble)")
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
    """Generates a markdown table for the top-(n) and bottom-(n) models by f1_score."""
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
        # Table header line (with simplified metric names)
        md_lines.append(
            "| Model | "
            + " | ".join(
                get_pretty_metric_label(col) for col in 
                ["f1_score", "accuracy", "precision", "recall", 
                 "execution_time_mean", "execution_time_sd", 
                 "eval_duration_mean", "prompt_eval_count_mean", "eval_count_mean"]
            )
            + " |"
        )
        # Table header alignment line
        md_lines.append(
            "|-------|" + "|".join(["------:"] * 9) + "|"
        )

        # Table rows
        for _, row in combined.iterrows():
            model_display = get_pretty_model_label(row['model'], use_linebreak=True)
            md_lines.append(
                f"| {model_display} "
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
# HTML table generator
# --------------------------------------------------------------------------
def generate_html_table(df, output_path, root_filename):
    """
    Generates an HTML file containing the same top-(n) and bottom-(n) model data
    as the Markdown table, using simplified model and metric names.
    """
    logger.info("Generating HTML table...")

    try:
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0)

        html_lines = []
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append("<meta charset='utf-8'>")
        html_lines.append("<title>Model Performance and Compute Summary (Top & Bottom)</title>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append("<h1>Model Performance and Compute Summary (Top & Bottom Models)</h1>")
        html_lines.append("<table border='1' style='border-collapse: collapse;'>")

        # Table header
        headers = [
            "Model",
            get_pretty_metric_label("f1_score"),
            get_pretty_metric_label("accuracy"),
            get_pretty_metric_label("precision"),
            get_pretty_metric_label("recall"),
            get_pretty_metric_label("execution_time_mean"),
            get_pretty_metric_label("execution_time_sd"),
            get_pretty_metric_label("eval_duration_mean"),
            get_pretty_metric_label("prompt_eval_count_mean"),
            get_pretty_metric_label("eval_count_mean")
        ]

        html_lines.append("<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")

        # Table rows
        for _, row in combined.iterrows():
            model_display = get_pretty_model_label(row['model'], use_linebreak=False)
            cells = [
                model_display,
                f"{row['f1_score']:.4f}",
                f"{row['accuracy']:.4f}",
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['execution_time_mean']:.4f}",
                f"{row['execution_time_sd']:.4f}",
                f"{row['eval_duration_mean']:.4f}",
                f"{row['prompt_eval_count_mean']:.2f}",
                f"{row['eval_count_mean']:.2f}"
            ]
            html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

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
    Generates a PDF file containing the same top-(n) and bottom-(n) model data
    as the Markdown table, using simplified model and metric names.
    """
    logger.info("Generating PDF table...")

    try:
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Model Performance and Compute Summary (Top & Bottom Models)", ln=True, align='C')
        pdf.ln(5)

        # Table headers
        pdf.set_font("Arial", 'B', 10)
        headers = [
            "Model",
            get_pretty_metric_label("f1_score"),
            get_pretty_metric_label("accuracy"),
            get_pretty_metric_label("precision"),
            get_pretty_metric_label("recall"),
            get_pretty_metric_label("execution_time_mean"),
            get_pretty_metric_label("execution_time_sd"),
            get_pretty_metric_label("eval_duration_mean"),
            get_pretty_metric_label("prompt_eval_count_mean"),
            get_pretty_metric_label("eval_count_mean")
        ]
        col_widths = [45, 13, 13, 13, 13, 16, 16, 17, 18, 18]  # Rough column widths

        # Print header row
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, txt=h, border=1, align='C')
        pdf.ln(8)

        # Table rows
        pdf.set_font("Arial", '', 9)
        for _, row in combined.iterrows():
            model_display = get_pretty_model_label(row['model'], use_linebreak=False)
            row_data = [
                model_display,
                f"{row['f1_score']:.4f}",
                f"{row['accuracy']:.4f}",
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['execution_time_mean']:.4f}",
                f"{row['execution_time_sd']:.4f}",
                f"{row['eval_duration_mean']:.4f}",
                f"{row['prompt_eval_count_mean']:.2f}",
                f"{row['eval_count_mean']:.2f}"
            ]
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
# Text report generator
# --------------------------------------------------------------------------
def generate_text_report(df, output_path, root_filename):
    """
    Outputs a plain-text summary of top/bottom model performance and additional 
    info that may not be in other files (e.g., total number of API calls, 
    number of failures, success rates, etc.).
    """
    logger.info("Generating text report...")

    try:
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
        top_models = df_sorted.head(TOP_OR_BOTTOM_CT)
        bottom_models = df_sorted.tail(TOP_OR_BOTTOM_CT)
        combined = pd.concat([top_models, bottom_models], axis=0)

        lines = []
        lines.append("========== TEXT REPORT SUMMARY ==========\n")

        lines.append("Top & Bottom Models (by F1 Score):")
        lines.append("----------------------------------\n")

        for _, row in combined.iterrows():
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

        # Additional metadata (e.g., total API calls, failures)
        lines.append("Additional Metadata:\n")
        # Attempt to compute stats if columns exist
        if 'api_calls' in df.columns:
            total_api_calls = df['api_calls'].sum()
            lines.append(f"  Total API Calls: {int(total_api_calls)}")

        if 'failures' in df.columns:
            total_failures = df['failures'].sum()
            lines.append(f"  Total Failures: {int(total_failures)}")
            if 'api_calls' in df.columns and total_api_calls > 0:
                # Compute success rate
                success_rate = (total_api_calls - total_failures) / total_api_calls * 100
                lines.append(f"  Success Rate: {success_rate:.2f}%")

        # Example: average F1 across all models
        avg_f1 = df['f1_score'].mean()
        lines.append(f"\n  Average {get_pretty_metric_label('f1_score')} (across entire dataset): {avg_f1:.4f}")

        lines.append("\n========== END OF TEXT REPORT ==========\n")

        output_filename = f"{root_filename}_summary_report.txt"
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

    # If you have extra columns for text report stats (like api_calls, failures),
    # copy them too if they exist:
    for col in ['api_calls', 'failures']:
        if col in df_raw.columns:
            df[col] = df_raw[col]

    # 7) Filter to only the base model names we want (keeping all prompt_type variants)
    df_filtered = filter_and_validate_models(df, ENSEMBLE_MODEL_LS)

    # 8) Plotting and table generation
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ROOT_FILENAME = f"stats_visualization_{ENSEMBLE_NAME}_{datetime_str}"
    
    sns.set_theme(style="whitegrid")

    # Existing outputs
    generate_markdown_table(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    create_performance_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    create_timing_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    create_token_plot(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)

    # New outputs (HTML, PDF, Text)
    generate_html_table(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    generate_pdf_table(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)
    generate_text_report(df_filtered, OUTPUT_ROOT_DIR, ROOT_FILENAME)

    logger.info("All visualization tasks completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)
