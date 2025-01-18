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
    Generates the 11 requested plots, with added numeric coercion for
    columns that represent compute times. Skips each plot if there's
    no valid numeric data, thereby avoiding blank/empty plots.
    """
    logging_custom("INFO", "Starting UPDATED visualizations...")

    # Make a subfolder for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Convert relevant columns to numeric (if they exist)
    numeric_cols = [
        'model_params',  # for param size
        'total_duration_median','total_duration_sd',
        'prompt_eval_duration_median','prompt_eval_duration_sd',
        'eval_duration_median','eval_duration_sd'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2) Create a numeric column for model_params to use in plots
    if 'model_params' in df.columns:
        df['model_params_numeric'] = df['model_params'].copy()
    else:
        df['model_params_numeric'] = np.nan

    #####################
    # PLOT 1
    #####################
    plot1_df = df.dropna(subset=['f1_score', 'model_name']).copy()
    if plot1_df.empty:
        logging_custom("WARNING", "Plot 1 skipped (no valid data).")
    else:
        plot1_df = plot1_df.sort_values('f1_score', ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=plot1_df, x='model_name', y='f1_score', order=plot1_df['model_name'])
        plt.xticks(rotation=30, ha="right")
        plt.title("Plot 1: F1 vs Model Name (descending F1)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot01_f1_vs_modelname_desc.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 2
    #####################
    plot2_df = df.dropna(subset=['f1_score','model_params_numeric']).copy()
    if plot2_df.empty:
        logging_custom("WARNING", "Plot 2 skipped (no valid data).")
    else:
        gp2 = plot2_df.groupby('model_params_numeric')['f1_score'].agg(['mean','std','count']).reset_index()
        gp2 = gp2.sort_values('model_params_numeric')
        plt.figure(figsize=(8,5))
        plt.errorbar(
            x=gp2['model_params_numeric'],
            y=gp2['mean'],
            yerr=gp2['std'],
            fmt='o-',
            capsize=4
        )
        plt.title("Plot 2: F1 vs Model Parameter Size (mean ± std)")
        plt.xlabel("Model Params (B)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot02_f1_vs_modelparams.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 3
    #####################
    plot3_df = df.dropna(subset=['f1_score','prompt_type']).copy()
    if plot3_df.empty:
        logging_custom("WARNING", "Plot 3 skipped (no valid data).")
    else:
        gp3 = plot3_df.groupby('prompt_type')['f1_score'].agg(['mean','std','count']).reset_index()
        gp3 = gp3.sort_values('mean', ascending=False)
        plt.figure(figsize=(6,5))
        plt.errorbar(
            x=range(len(gp3)),
            y=gp3['mean'],
            yerr=gp3['std'],
            fmt='o',
            capsize=4
        )
        plt.xticks(range(len(gp3)), gp3['prompt_type'], rotation=20)
        plt.title("Plot 3: F1 vs Prompt Type (mean ± std)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot03_f1_vs_prompttype.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 4
    #####################
    plot4_df = df.dropna(subset=['f1_score','model_params_numeric','model_quantization']).copy()
    plot4_df = plot4_df[plot4_df['model_quantization'].isin(['q4_k_m','fp16'])]
    if plot4_df.empty:
        logging_custom("WARNING", "Plot 4 skipped (no valid q4_k_m/fp16 data).")
    else:
        plot4_df = plot4_df.sort_values('model_params_numeric')
        plt.figure(figsize=(8,5))
        sns.pointplot(
            data=plot4_df,
            x='model_params_numeric',
            y='f1_score',
            hue='model_quantization',
            ci='sd',
            markers='o'
        )
        plt.title("Plot 4: F1 vs Model Param Size (q4_k_m vs fp16)")
        plt.xlabel("Model Params (B)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot04_f1_vs_paramsize_q4km_fp16.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 5
    #####################
    plot5_df = df.dropna(subset=['f1_score','total_duration_median','total_duration_sd']).copy()
    if plot5_df.empty:
        logging_custom("WARNING", "Plot 5 skipped (no valid total_duration data).")
    else:
        plot5_df = plot5_df.sort_values('total_duration_median')
        plt.figure(figsize=(8,5))
        plt.errorbar(
            x=plot5_df['total_duration_median'],
            y=plot5_df['f1_score'],
            xerr=plot5_df['total_duration_sd'],
            fmt='o',
            capsize=4
        )
        plt.title("Plot 5: F1 vs total_duration_median (+/- total_duration_sd)")
        plt.xlabel("Total Duration (median)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot05_f1_vs_totalduration.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 6
    #####################
    plot6_df = df.dropna(subset=['f1_score','prompt_eval_duration_median','prompt_eval_duration_sd']).copy()
    if plot6_df.empty:
        logging_custom("WARNING", "Plot 6 skipped (no valid prompt_eval_duration data).")
    else:
        plot6_df = plot6_df.sort_values('prompt_eval_duration_median')
        plt.figure(figsize=(8,5))
        plt.errorbar(
            x=plot6_df['prompt_eval_duration_median'],
            y=plot6_df['f1_score'],
            xerr=plot6_df['prompt_eval_duration_sd'],
            fmt='o',
            capsize=4
        )
        plt.title("Plot 6: F1 vs prompt_eval_duration_median (+/- prompt_eval_duration_sd)")
        plt.xlabel("Prompt Eval Duration (median)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot06_f1_vs_prompt_eval_duration.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 7
    #####################
    plot7_df = df.dropna(subset=['f1_score','eval_duration_median','eval_duration_sd']).copy()
    if plot7_df.empty:
        logging_custom("WARNING", "Plot 7 skipped (no valid eval_duration data).")
    else:
        plot7_df = plot7_df.sort_values('eval_duration_median')
        plt.figure(figsize=(8,5))
        plt.errorbar(
            x=plot7_df['eval_duration_median'],
            y=plot7_df['f1_score'],
            xerr=plot7_df['eval_duration_sd'],
            fmt='o',
            capsize=4
        )
        plt.title("Plot 7: F1 vs eval_duration_median (+/- eval_duration_sd)")
        plt.xlabel("Eval Duration (median)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot07_f1_vs_eval_duration.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 8
    #####################
    plot8_df = df.dropna(subset=['f1_score','model_params_numeric']).copy()
    if plot8_df.empty:
        logging_custom("WARNING", "Plot 8 skipped (no valid param size data).")
    else:
        plot8_df = plot8_df.sort_values('model_params_numeric')
        plt.figure(figsize=(8,5))
        plt.scatter(plot8_df['model_params_numeric'], plot8_df['f1_score'], alpha=0.7)
        plt.title("Plot 8: F1 vs Param Size")
        plt.xlabel("Model Params (B)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot08_f1_vs_paramsize_scatter.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 9
    #####################
    plot9_df = df.dropna(subset=['f1_score','prompt_eval_duration_median']).copy()
    if plot9_df.empty:
        logging_custom("WARNING", "Plot 9 skipped (no valid prompt_eval_duration_median data).")
    else:
        plot9_df = plot9_df.sort_values('prompt_eval_duration_median')
        plt.figure(figsize=(8,5))
        plt.scatter(plot9_df['prompt_eval_duration_median'], plot9_df['f1_score'], alpha=0.7)
        plt.title("Plot 9: F1 vs prompt_eval_duration_median")
        plt.xlabel("Prompt Eval Duration (median)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot09_f1_vs_prompt_eval_scatter.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 10
    #####################
    plot10_df = df.dropna(subset=['f1_score','eval_duration_median']).copy()
    if plot10_df.empty:
        logging_custom("WARNING", "Plot 10 skipped (no valid eval_duration_median data).")
    else:
        plot10_df = plot10_df.sort_values('eval_duration_median')
        plt.figure(figsize=(8,5))
        plt.scatter(plot10_df['eval_duration_median'], plot10_df['f1_score'], alpha=0.7)
        plt.title("Plot 10: F1 vs eval_duration_median")
        plt.xlabel("Eval Duration (median)")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot10_f1_vs_evalduration_scatter.png"), dpi=150)
        plt.close()

    #####################
    # PLOT 11
    #####################
    plot11_df = df.dropna(subset=['f1_score','total_duration_median']).copy()
    if plot11_df.empty:
        logging_custom("WARNING", "Plot 11 skipped (no valid total_duration_median data).")
    else:
        plot11_df = plot11_df.sort_values('f1_score')
        plt.figure(figsize=(8,5))
        plt.scatter(plot11_df['f1_score'], plot11_df['total_duration_median'], alpha=0.7)
        plt.title("Plot 11: F1 (x-axis) vs total_duration_median (y-axis)")
        plt.xlabel("F1 Score")
        plt.ylabel("Total Duration (median)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "plot11_f1x_vs_totaldurationy.png"), dpi=150)
        plt.close()

    logging_custom("INFO", f"All updated plots saved to {plots_dir}.")



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

    # 2. Create the new visualizations
    create_visualizations(df, OUTPUT_DIR)

    logging_custom("INFO", "All done!")


if __name__ == "__main__":
    main()
