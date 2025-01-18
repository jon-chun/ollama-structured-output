#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import logging
from datetime import datetime

# --------------------------------------------------------------------
# 1. Global Constants
# --------------------------------------------------------------------

# EVALUATION_RESULTS_DIR = 'evaluation_results_long_20250110'
EVALUATION_RESULTS_DIR = 'evaluation_results_short_20250108'

ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)
OUTPUT_DIR = os.path.join('aggregate_reports', os.path.basename(EVALUATION_RESULTS_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILENAME = "aggregate_model_reports.csv"


# --------------------------------------------------------------------
# 2. Setup Logging
# --------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"log_aggregate_reports_{timestamp}.txt"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # capture all levels

# File handler (DEBUG-level messages go to file)
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)

# Console handler (INFO-level messages to console)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Common formatter
formatter = logging.Formatter(
    '[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

def logging_custom(level: str, message: str):
    """
    A single custom logging function that routes messages based on `level`.
    Valid levels: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    level = level.upper().strip()
    if level == 'DEBUG':
        logger.debug(message)
    elif level == 'INFO':
        logger.info(message)
    elif level == 'WARNING':
        logger.warning(message)
    elif level == 'ERROR':
        logger.error(message)
    elif level == 'CRITICAL':
        logger.critical(message)
    else:
        logger.debug(f"[INVALID LEVEL: {level}] {message}")


# --------------------------------------------------------------------
# 3. Utility Functions
# --------------------------------------------------------------------
def extract_model_metadata(model_dir_name):
    """
    Extract model size and quantization info from directory name.
    Now returns renamed fields.
    """
    logging_custom("DEBUG", f"extract_model_metadata => directory: {model_dir_name}")

    param_match = re.search(r'(\d+)b', model_dir_name.lower())
    quant_match = re.search(r'(q4_k_m|fp16)', model_dir_name.lower())

    params = param_match.group(1) if param_match else "unknown"
    quant = quant_match.group(1) if quant_match else "unknown"

    logging_custom("DEBUG", f" -> model_params={params}, model_quantization={quant}")
    return {
        'model_params': params,  # Renamed from model_params_b
        'model_quantization': quant  # Renamed from quantization
    }

def clean_model_name(model_name: str) -> str:
    """
    Convert a model name to an OS-friendly format consistently.
    For example: 'llama3.2:1b-instruct-q4_K_M' -> 'llama3_2_1b_instruct_q4_k_m'
    """
    cleaned = model_name.strip().lower()
    # Replace colons and periods with underscore
    cleaned = re.sub(r'[:.]+', '_', cleaned)
    # Replace other punctuation/whitespace with underscores
    cleaned = re.sub(r'[^\w]+', '_', cleaned)
    # Collapse multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Strip leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned

def get_txt_for_json_filename(json_filename: str) -> str:
    """
    Given a JSON filename, derive the matching TXT filename.
    If JSON is prefixed with 'metrics_', then the TXT is prefixed with 'report_'.
    Otherwise, fallback to the same base name.
    """
    base_name, _ = os.path.splitext(json_filename)
    if base_name.startswith("metrics_"):
        txt_name = "report_" + base_name[len("metrics_"):]
    else:
        txt_name = base_name
    return txt_name + ".txt"

# --------------------------------------------------------------------
# 4. Parsing Functions
# --------------------------------------------------------------------
def parse_report_json(json_path):
    """
    Parse a JSON file and return a dictionary of relevant fields.
    """
    logging_custom("DEBUG", f"Parsing JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def parse_report_txt(txt_path):
    """
    Parse a TXT file with summary metrics (example lines: 'Accuracy: 53.00%', etc.).
    Return a dict of parsed data.
    """
    logging_custom("DEBUG", f"Parsing TXT file: {txt_path}")
    summary = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line_stripped = line.strip()
        # Example: "Accuracy: 53.00%"
        if line_stripped.startswith("Accuracy:"):
            val = line_stripped.split("Accuracy:")[-1].strip().replace('%', '')
            try:
                summary['txt_accuracy'] = float(val)
            except ValueError:
                summary['txt_accuracy'] = None

        # Example: "AUC-ROC: 0.4699"
        if "AUC-ROC:" in line_stripped:
            val = line_stripped.split("AUC-ROC:")[-1].strip()
            try:
                summary['txt_auc_roc'] = float(val)
            except ValueError:
                summary['txt_auc_roc'] = None

    return summary


def process_report_files(json_path, txt_path):
    """
    Process a JSON report and a TXT file, returning a combined dict of stats.
    Now includes additional duration statistics.
    """
    logging_custom("DEBUG", f"process_report_files => JSON: {json_path}, TXT: {txt_path}")
    try:
        jdata = parse_report_json(json_path)
    except Exception as e:
        logging_custom("ERROR", f"Error reading JSON: {json_path} => {e}")
        return None

    try:
        tdata = parse_report_txt(txt_path)
    except Exception as e:
        logging_custom("ERROR", f"Error reading TXT: {txt_path} => {e}")
        tdata = {}

    combined = {}

    # Flatten some top-level JSON fields with renamed columns
    combined['model_name'] = jdata.get('model_name')
    combined['prompt_type'] = jdata.get('prompt_type')
    combined['total_attempts'] = jdata.get('total_attempts')
    combined['successful_attempts'] = jdata.get('successful_attempts')
    combined['failed_attempts'] = jdata.get('failed_attempts')
    combined['timeout_attempts'] = jdata.get('timeout_attempts')
    
    # Rename execution time metrics
    combined['execution_time_mean'] = jdata.get('execution_time_mean')
    combined['execution_time_median'] = jdata.get('execution_time_median')
    combined['execution_time_sd'] = jdata.get('execution_time_sd')
    combined['prediction_accuracy'] = jdata.get('prediction_accuracy')
    combined['auc_roc'] = jdata.get('auc_roc')

    # Gather raw durations from jdata => meta_data (if present)
    meta_data = jdata.get('meta_data', [])
    total_durations = []
    load_durations = []
    prompt_eval_durations = []
    eval_durations = []
    
    for entry in meta_data:
        if isinstance(entry, dict):
            total_durations.append(entry.get('total_duration'))
            load_durations.append(entry.get('load_duration'))
            prompt_eval_durations.append(entry.get('prompt_eval_duration'))
            eval_durations.append(entry.get('eval_duration'))
    
    # Helper to get mean, median, sd
    def calculate_stats(data):
        clean_data = [x for x in data if x is not None]
        if clean_data:
            return {
                'mean': np.mean(clean_data),
                'median': np.median(clean_data),
                'sd': np.std(clean_data) if len(clean_data) > 1 else 0
            }
        return {'mean': None, 'median': None, 'sd': None}
    
    total_stats = calculate_stats(total_durations)
    load_stats = calculate_stats(load_durations)
    prompt_eval_stats = calculate_stats(prompt_eval_durations)
    eval_stats = calculate_stats(eval_durations)
    
    combined['total_duration_mean'] = total_stats['mean']
    combined['total_duration_median'] = total_stats['median']
    combined['total_duration_sd'] = total_stats['sd']
    
    combined['load_duration_mean'] = load_stats['mean']
    combined['load_duration_median'] = load_stats['median']
    combined['load_duration_sd'] = load_stats['sd']
    
    combined['prompt_eval_duration_mean'] = prompt_eval_stats['mean']
    combined['prompt_eval_duration_median'] = prompt_eval_stats['median']
    combined['prompt_eval_duration_sd'] = prompt_eval_stats['sd']
    
    combined['eval_duration_mean'] = eval_stats['mean']
    combined['eval_duration_median'] = eval_stats['median']
    combined['eval_duration_sd'] = eval_stats['sd']

    # Confusion matrix
    confusion = jdata.get('confusion_matrix', {})
    tp = confusion.get('tp', 0)
    tn = confusion.get('tn', 0)
    fp = confusion.get('fp', 0)
    fn = confusion.get('fn', 0)
    combined['true_positives'] = tp
    combined['true_negatives'] = tn
    combined['false_positives'] = fp
    combined['false_negatives'] = fn

    # Derived classification metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    combined['precision'] = precision
    combined['recall'] = recall
    combined['f1_score'] = f1_score

    # Add text-based stats
    combined['txt_accuracy'] = tdata.get('txt_accuracy')
    combined['txt_auc_roc'] = tdata.get('txt_auc_roc')

    return combined


# --------------------------------------------------------------------
# 5. Core Aggregation
# --------------------------------------------------------------------
def aggregate_reports():
    """
    Crawl the directory tree looking for 'reports' directories,
    then find .json + .txt file pairs. Return a DataFrame of all data.
    Includes filtering for the required prompt types: COT, COT_NSHOT, SYSTEM1.
    """
    logging_custom("INFO", f"Starting directory crawl from: {ROOT_DIR}")

    if not os.path.exists(ROOT_DIR):
        logging_custom("ERROR", f"Evaluation directory not found: {ROOT_DIR}")
        return pd.DataFrame()

    aggregated_rows = []

    for root, dirs, files in os.walk(ROOT_DIR):
        logging_custom("DEBUG", f"Scanning directory: {root}")
        logging_custom("DEBUG", f" - Subdirs: {dirs}")
        logging_custom("DEBUG", f" - Files: {files}")

        # Check if current folder is 'reports'
        if os.path.basename(root) == 'reports':
            logging_custom("INFO", f"Found a 'reports' directory: {root}")

            # The parent of 'reports' is presumably the model directory
            model_dir_name = os.path.basename(os.path.dirname(root))
            logging_custom("DEBUG", f" => parent model_dir_name: {model_dir_name}")

            # Here we pull in metadata about the model
            model_meta = {
                'model_dir': model_dir_name,
                **extract_model_metadata(model_dir_name)
            }

            # Gather JSON files
            json_files = [f for f in files if f.endswith('.json')]
            logging_custom("DEBUG", f" => Found JSON files: {json_files}")

            for jf in json_files:
                # Attempt to find the matching text file
                tf = get_txt_for_json_filename(jf)
                logging_custom("DEBUG", f"Looking for corresponding TXT file: {tf}")

                if tf in files:
                    json_path = os.path.join(root, jf)
                    txt_path = os.path.join(root, tf)
                    logging_custom("INFO", f"Processing pair => JSON: {jf}, TXT: {tf}")

                    combined_stats = process_report_files(json_path, txt_path)
                    if combined_stats:
                        # Add model metadata
                        combined_stats.update(model_meta)
                        aggregated_rows.append(combined_stats)
                        logging_custom("INFO", f"Added data for {jf}")
                else:
                    logging_custom("WARNING", f"No matching TXT for JSON: {jf} in {root}")

    # Create initial DataFrame
    df = pd.DataFrame(aggregated_rows)
    
    if df.empty:
        return df
        
    # Step 1: Group by model_name and filter out incomplete sets
    valid_models_data = []
    
    # Get unique model names
    model_names = df['model_name'].unique()
    
    # Required prompt types
    required_prompts = {'PromptType.COT', 'PromptType.COT_NSHOT', 'PromptType.SYSTEM1'}
    
    for model in model_names:
        model_df = df[df['model_name'] == model]
        model_prompts = set(model_df['prompt_type'].unique())
        
        # Check if model has all required prompt types
        if model_prompts >= required_prompts:
            # Collect exactly one row per required prompt
            model_rows = []
            for prompt_type in required_prompts:
                prompt_rows = model_df[model_df['prompt_type'] == prompt_type]
                if not prompt_rows.empty:
                    model_rows.append(prompt_rows.iloc[0])  # first occurrence

            if len(model_rows) == 3:
                valid_models_data.extend(model_rows)
                logging_custom("INFO", f"Added complete set of prompts for model: {model}")
        else:
            missing = required_prompts - model_prompts
            logging_custom("INFO", f"Skipping model {model} - missing prompt types: {missing}")
    
    final_df = pd.DataFrame(valid_models_data)
    
    logging_custom("INFO", f"Final number of models: {len(final_df['model_name'].unique())}")
    logging_custom("INFO", f"Final number of rows: {len(final_df)}")
    
    return final_df


# --------------------------------------------------------------------
# 6. Visualization & Reporting
# --------------------------------------------------------------------
def create_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Generates 10 key comparisons and 2 plots each (20 total), then writes
    a markdown summary to 'aggregate_summary.md' in output_dir.
    """

    logging_custom("INFO", "Starting visualizations...")

    # Make a subfolder for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    commentary_list = []
    
    # Helper function to format filenames systematically
    def plot_filename(comp_num, sublabel):
        return os.path.join(plots_dir, f"comparison{comp_num:02d}_plot{sublabel}.png")

    # ---------- Comparison 1: Accuracy by Model + Prompt Type ----------
    comparison_num = 1
    metric = "prediction_accuracy"

    # A) Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="model_name",
        y=metric,
        hue="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: {metric} by Model + Prompt Type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Strip Plot
    plt.figure(figsize=(8, 5))
    sns.stripplot(
        data=df,
        x="model_name",
        y=metric,
        hue="prompt_type",
        dodge=True
    )
    plt.title(f"Comparison {comparison_num}B: {metric} (Strip) by Model + Prompt Type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_1 = (
        f"**Comparison {comparison_num}: Accuracy by Model/Prompt**\n"
        f"This reveals how `prediction_accuracy` differs across `model_name` and `prompt_type`. "
        f"Bar and strip plots illustrate the distribution and relative performance."
    )
    commentary_list.append(comment_1)

    # ---------- Comparison 2: F1 Score by Model + Prompt Type ----------
    comparison_num = 2
    metric = "f1_score"

    # A) Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="model_name",
        y=metric,
        hue="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: F1 Score by Model + Prompt Type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Strip Plot
    plt.figure(figsize=(8, 5))
    sns.stripplot(
        data=df,
        x="model_name",
        y=metric,
        hue="prompt_type",
        dodge=True
    )
    plt.title(f"Comparison {comparison_num}B: F1 Score (Strip) by Model + Prompt Type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_2 = (
        f"**Comparison {comparison_num}: F1 by Model/Prompt**\n"
        f"Shows how different prompt types and models fare with respect to F1, "
        f"balancing precision and recall."
    )
    commentary_list.append(comment_2)

    # ---------- Comparison 3: Execution Time (Mean) vs. Accuracy ----------
    comparison_num = 3

    # A) Scatter: execution_time_mean vs prediction_accuracy
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="execution_time_mean",
        y="prediction_accuracy",
        hue="model_name",
        style="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: Mean Execution Time vs. Accuracy")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Scatter with regression
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x="execution_time_mean",
        y="prediction_accuracy",
        scatter=True,
        scatter_kws={"alpha":0.5},
        line_kws={"color":"red"}
    )
    plt.title(f"Comparison {comparison_num}B: Regression of Execution Time vs. Accuracy")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_3 = (
        f"**Comparison {comparison_num}: Exec Time vs. Accuracy**\n"
        f"Visualizes trade-offs between speed and correctness. The second chart includes a regression line."
    )
    commentary_list.append(comment_3)

    # ---------- Comparison 4: Precision vs. Recall ----------
    comparison_num = 4

    # A) Scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="precision",
        y="recall",
        hue="model_name",
        style="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: Precision vs. Recall by Model + Prompt")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Scatter colored by F1
    plt.figure(figsize=(8, 5))
    scatter = sns.scatterplot(
        data=df,
        x="precision",
        y="recall",
        hue="f1_score",
        palette="coolwarm",
        s=100
    )
    plt.title(f"Comparison {comparison_num}B: Precision vs. Recall, colored by F1")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_4 = (
        f"**Comparison {comparison_num}: Precision vs. Recall**\n"
        f"Examines the trade-off between precision and recall. "
        f"The second plot color-codes points by `f1_score`."
    )
    commentary_list.append(comment_4)

    # ---------- Comparison 5: AUC-ROC by Model + Prompt Type ----------
    comparison_num = 5
    metric = "auc_roc"

    # A) Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="model_name",
        y=metric,
        hue="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: AUC-ROC by Model + Prompt")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Box Plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="model_name",
        y=metric,
        hue="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}B: AUC-ROC Distribution by Model + Prompt")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_5 = (
        f"**Comparison {comparison_num}: AUC-ROC by Model/Prompt**\n"
        f"Shows how well the model ranks positive vs. negative across different prompts."
    )
    commentary_list.append(comment_5)

    # ---------- Comparison 6: Median vs. Mean Execution Time ----------
    comparison_num = 6

    # A) Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="execution_time_mean",
        y="execution_time_median",
        hue="model_name",
        style="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: Execution Time Median vs. Mean")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Plot with diagonal reference line
    plt.figure(figsize=(8, 5))
    ax = sns.scatterplot(
        data=df,
        x="execution_time_mean",
        y="execution_time_median",
        hue="prompt_type"
    )
    max_val = max(
        df["execution_time_mean"].dropna().max(),
        df["execution_time_median"].dropna().max()
    )
    plt.plot([0, max_val], [0, max_val], "r--", label="y=x reference")
    plt.title(f"Comparison {comparison_num}B: Mean vs. Median with y=x line")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_6 = (
        f"**Comparison {comparison_num}: Mean vs. Median Exec Time**\n"
        f"Points far from the y=x line indicate skew or outliers in runtime."
    )
    commentary_list.append(comment_6)

    # ---------- Comparison 7: Distribution of Execution Time SD ----------
    comparison_num = 7
    metric = "execution_time_sd"

    # A) Bar Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="model_name",
        y=metric,
        hue="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: Execution Time SD by Model + Prompt")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Violin Plot
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=df,
        x="prompt_type",
        y=metric,
        hue="model_name",
        split=True
    )
    plt.title(f"Comparison {comparison_num}B: Distribution of Execution Time SD")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_7 = (
        f"**Comparison {comparison_num}: Execution Time SD**\n"
        f"Larger SD implies more variance in per-attempt runtimes."
    )
    commentary_list.append(comment_7)

    # ---------- Comparison 8: total_attempts vs. Accuracy ----------
    comparison_num = 8

    # A) Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="total_attempts",
        y="prediction_accuracy",
        hue="model_name",
        style="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: total_attempts vs. Accuracy")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="model_name",
        y="prediction_accuracy",
        hue="total_attempts"
    )
    plt.title(f"Comparison {comparison_num}B: Accuracy grouped by total_attempts and Model")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_8 = (
        f"**Comparison {comparison_num}: total_attempts vs. Accuracy**\n"
        f"Verifies how the sample size per model relates to the final accuracy."
    )
    commentary_list.append(comment_8)

    # ---------- Comparison 9: Load/Prompt/Eval Durations ----------
    comparison_num = 9

    df_melted = df.melt(
        id_vars=["model_name", "prompt_type"],
        value_vars=[
            "load_duration_mean",
            "prompt_eval_duration_mean",
            "eval_duration_mean"
        ],
        var_name="duration_type",
        value_name="duration_value"
    )

    # A) Bar Chart
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df_melted,
        x="model_name",
        y="duration_value",
        hue="duration_type"
    )
    plt.title(f"Comparison {comparison_num}A: Duration Means by Model + Prompt")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Catplot by prompt_type & model_name
    g = sns.catplot(
        data=df_melted,
        x="prompt_type",
        y="duration_value",
        hue="duration_type",
        col="model_name",
        kind="bar",
        col_wrap=3,
        height=4,
        aspect=1.2
    )
    g.fig.suptitle(f"Comparison {comparison_num}B: Load/Prompt/Eval Duration by Model + Prompt")
    g.fig.tight_layout()
    # Must save via g.fig
    g.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close("all")

    comment_9 = (
        f"**Comparison {comparison_num}: Load/Prompt/Eval Durations**\n"
        f"Shows how time is divided among loading, prompt evaluation, and final eval for each model & prompt."
    )
    commentary_list.append(comment_9)

    # ---------- Comparison 10: txt_accuracy vs. prediction_accuracy ----------
    comparison_num = 10

    # A) Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="prediction_accuracy",
        y="txt_accuracy",
        hue="model_name",
        style="prompt_type"
    )
    plt.title(f"Comparison {comparison_num}A: Numeric vs. Text Accuracy")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "A"), dpi=150)
    plt.close()

    # B) Bar Plot of differences
    df["accuracy_diff"] = df["txt_accuracy"] - df["prediction_accuracy"]
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="model_name",
        y="accuracy_diff",
        hue="prompt_type"
    )
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f"Comparison {comparison_num}B: (txt_accuracy - prediction_accuracy) by Model + Prompt")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plot_filename(comparison_num, "B"), dpi=150)
    plt.close()

    comment_10 = (
        f"**Comparison {comparison_num}: txt_accuracy vs. numeric accuracy**\n"
        f"Checks whether textual accuracy from the report matches numeric calculations."
    )
    commentary_list.append(comment_10)

    # Finally: Summarize these findings in a markdown file
    md_path = os.path.join(output_dir, "aggregate_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Aggregate Summary of Key Comparisons\n\n")
        f.write("Below are 10 major comparisons using the aggregated summary data.\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n")
        for i, ctext in enumerate(commentary_list, start=1):
            f.write(f"- [Comparison {i}](#comparison-{i})\n")
        f.write("\n")
        
        # Write each comparison's discussion
        for i, ctext in enumerate(commentary_list, start=1):
            f.write(f"## Comparison {i}\n\n")
            f.write(f"{ctext}\n\n")

    logging_custom("INFO", f"Visualizations saved in {plots_dir}, summary in {md_path}")


# --------------------------------------------------------------------
# 7. Main Execution
# --------------------------------------------------------------------
def main():
    logging_custom("INFO", "Starting report aggregation...")
    df = aggregate_reports()

    if df.empty:
        logging_custom("WARNING", "No valid reports were found.")
        logging_custom("WARNING", "No data to save or plot. Exiting.")
        return

    # 1. Save the aggregated DataFrame
    output_csv = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(output_csv, index=False)
    logging_custom("INFO", f"Aggregated {len(df)} rows saved to: {output_csv}")

    # 2. Create the visualizations & markdown summary
    create_visualizations(df, OUTPUT_DIR)

    logging_custom("INFO", "All done!")


if __name__ == "__main__":
    main()
