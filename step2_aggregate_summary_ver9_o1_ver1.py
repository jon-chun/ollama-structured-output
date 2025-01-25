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
EVALUATION_RESULTS_DIR = 'evaluation_results_long'

ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)
OUTPUT_DIR = os.path.join('aggregate_reports', os.path.basename(EVALUATION_RESULTS_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_FILENAME = f"aggregate_model_reports_o1.csv"

# Columns we want in the final CSV, in the specified order.
OUTPUT_FILE_DESIRED_COLUMNS = [
    "model_name",
    "prompt_type",
    "total_attempts",
    "successful_attempts",
    "failed_attempts",
    "timeout_attempts",
    "execution_time_mean",
    "execution_time_median",
    "execution_time_sd",
    "prediction_accuracy",
    "auc_roc",
    "txt_accuracy",
    "txt_auc_roc",
    "total_duration_mean",
    "total_duration_median",
    "total_duration_sd",
    "load_duration_mean",
    "load_duration_median",
    "load_duration_sd",
    "prompt_eval_duration_mean",
    "prompt_eval_duration_median",
    "prompt_eval_duration_sd",
    "eval_duration_mean",
    "eval_duration_median",
    "eval_duration_sd",
    "prompt_eval_count_mean",
    "prompt_eval_count_median",
    "prompt_eval_count_sd",
    "eval_count_mean",
    "eval_count_median",
    "eval_count_sd",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1_score",
    "model_dir",
    "model_params",
    "model_quantization",
    "total_duration_sec_missing_count",
    "load_duration_sec_missing_count",
    "prompt_eval_duration_sec_missing_count",
    "eval_duration_sec_missing_count",
    "python_api_duration_sec_missing_count",
    "confidence_txt_missing_count",
    "prompt_eval_count_missing_count",
    "eval_count_missing_count"
]

# --------------------------------------------------------------------
# 2. Setup Logging
# --------------------------------------------------------------------
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
    quant_match = re.search(r'(q4_k_m|fp16|q8_0|q4_k_g|q4_k_m)', model_dir_name.lower())
    # Extended the regex for potential additional quantization patterns if needed

    params = param_match.group(1) if param_match else "unknown"
    quant = quant_match.group(1) if quant_match else "unknown"

    logging_custom("DEBUG", f" -> model_params={params}, model_quantization={quant}")
    return {
        'model_params': params,  
        'model_quantization': quant
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

def calculate_stats(data):
    """
    Helper function to calculate mean, median, and standard deviation
    Returns a dictionary with keys: 'mean', 'median', 'sd'
    """
    clean_data = [x for x in data if x is not None]
    if clean_data:
        return {
            'mean': np.mean(clean_data),
            'median': np.median(clean_data),
            'sd': np.std(clean_data) if len(clean_data) > 1 else 0
        }
    return {'mean': None, 'median': None, 'sd': None}


# --------------------------------------------------------------------
# 4. Parsing Functions
# --------------------------------------------------------------------
def parse_report_json(json_path):
    """
    Parse a JSON file and return a dictionary of the entire structure.
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
    Expanded to handle additional stats and missing count logic.
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
    # Some may not exist if the JSON structure is different
    combined['model_name'] = jdata.get('evaluation', {}).get('model', None) or jdata.get('model_name')
    combined['prompt_type'] = jdata.get('evaluation', {}).get('prompt_type', None) or jdata.get('prompt_type')
    combined['total_attempts'] = jdata.get('total_attempts')
    combined['successful_attempts'] = jdata.get('successful_attempts')
    combined['failed_attempts'] = jdata.get('failed_attempts')
    combined['timeout_attempts'] = jdata.get('timeout_attempts')
    
    # Execution times
    combined['execution_time_mean'] = jdata.get('avg_execution_time')
    combined['execution_time_median'] = jdata.get('median_execution_time')
    combined['execution_time_sd'] = jdata.get('sd_execution_time')
    combined['prediction_accuracy'] = jdata.get('prediction_accuracy')
    combined['auc_roc'] = jdata.get('auc_roc')

    # Text-based stats
    combined['txt_accuracy'] = tdata.get('txt_accuracy')
    combined['txt_auc_roc'] = tdata.get('txt_auc_roc')

    # Confusion Matrix
    confusion = jdata.get('confusion_matrix', {})
    tp = confusion.get('tp', 0)
    tn = confusion.get('tn', 0)
    fp = confusion.get('fp', 0)
    fn = confusion.get('fn', 0)
    combined['true_positives'] = tp
    combined['true_negatives'] = tn
    combined['false_positives'] = fp
    combined['false_negatives'] = fn

    # Derived metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    combined['precision'] = precision
    combined['recall'] = recall
    combined['f1_score'] = f1_score

    # --------------------------------------------------------------------
    # Additional Fields: prompt_eval_count, eval_count, durations, etc.
    # We unify meta_data to a list, even if it is a single dict in the JSON
    # --------------------------------------------------------------------
    meta_data = jdata.get('meta_data', [])
    if isinstance(meta_data, dict):
        meta_data = [meta_data]

    total_durations = []
    load_durations = []
    prompt_eval_durations = []
    eval_durations = []
    python_api_durations = []  # if available
    prompt_eval_counts = []
    eval_counts = []

    # We'll also track row-by-row missingness
    # (But each JSON presumably has only one meta_data dict in typical usage.)
    for entry in meta_data:
        if isinstance(entry, dict):
            total_durations.append(entry.get('total_duration'))
            load_durations.append(entry.get('load_duration'))
            prompt_eval_durations.append(entry.get('prompt_eval_duration'))
            eval_durations.append(entry.get('eval_duration'))
            python_api_durations.append(entry.get('python_api_duration'))  # Might be missing
            prompt_eval_counts.append(entry.get('prompt_eval_count'))
            eval_counts.append(entry.get('eval_count'))

    # Duration stats
    total_stats = calculate_stats(total_durations)
    load_stats = calculate_stats(load_durations)
    prompt_eval_stats = calculate_stats(prompt_eval_durations)
    eval_stats = calculate_stats(eval_durations)
    python_api_stats = calculate_stats(python_api_durations)

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

    # Prompt/Eval count stats
    prompt_eval_count_stats = calculate_stats(prompt_eval_counts)
    eval_count_stats = calculate_stats(eval_counts)

    combined['prompt_eval_count_mean'] = prompt_eval_count_stats['mean']
    combined['prompt_eval_count_median'] = prompt_eval_count_stats['median']
    combined['prompt_eval_count_sd'] = prompt_eval_count_stats['sd']

    combined['eval_count_mean'] = eval_count_stats['mean']
    combined['eval_count_median'] = eval_count_stats['median']
    combined['eval_count_sd'] = eval_count_stats['sd']

    # --------------------------------------------------------------------
    # Missing Count Columns
    # For each row, 1 if the field is missing/None, else 0
    # We'll treat "missing" as either no key or key == None
    # Since we might have arrays above, check the first element
    # --------------------------------------------------------------------
    def is_missing_first_value(values_list):
        if not values_list or values_list[0] is None:
            return 1
        return 0

    combined['total_duration_sec_missing_count'] = is_missing_first_value(total_durations)
    combined['load_duration_sec_missing_count'] = is_missing_first_value(load_durations)
    combined['prompt_eval_duration_sec_missing_count'] = is_missing_first_value(prompt_eval_durations)
    combined['eval_duration_sec_missing_count'] = is_missing_first_value(eval_durations)
    combined['python_api_duration_sec_missing_count'] = is_missing_first_value(python_api_durations)

    # Check "decision.confidence" or "response.confidence"; sample shows "decision" -> "confidence"
    decision_conf = None
    if 'decision' in jdata and isinstance(jdata['decision'], dict):
        decision_conf = jdata['decision'].get('confidence', None)
    # If you want to handle other structures, do so similarly
    combined['confidence_txt_missing_count'] = 1 if (decision_conf is None) else 0

    combined['prompt_eval_count_missing_count'] = is_missing_first_value(prompt_eval_counts)
    combined['eval_count_missing_count'] = is_missing_first_value(eval_counts)

    return combined


# --------------------------------------------------------------------
# 5. Core Aggregation
# --------------------------------------------------------------------
def aggregate_reports():
    """
    Crawl the directory tree looking for 'reports' directories,
    then find .json + .txt file pairs. Return a DataFrame of all data.
    Includes filtering for complete prompt-type sets and duplicate removal.
    Adds new columns in final DataFrame.
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
        logging_custom("WARNING", "No data extracted from JSON files. Returning empty DataFrame.")
        return df

    # --------------------------------------------------------------------
    # Filter: We only keep models that have all required prompt types
    # --------------------------------------------------------------------
    required_prompts = {'PromptType.COT', 'PromptType.COT_NSHOT', 'PromptType.SYSTEM1'}
    final_rows = []
    model_names = df['model_name'].dropna().unique()

    for model in model_names:
        model_df = df[df['model_name'] == model]
        model_prompts = set(model_df['prompt_type'].dropna().unique())

        missing_prompts = required_prompts - model_prompts
        if len(missing_prompts) == 0:
            # model has all required prompts
            # For each required prompt, we can pick the first row or do more logic
            for pt in required_prompts:
                subset = model_df[model_df['prompt_type'] == pt]
                if not subset.empty:
                    final_rows.append(subset.iloc[0].to_dict())
            logging_custom("INFO", f"Model {model} includes all required prompt types.")
        else:
            logging_custom("INFO", f"Skipping model {model}, missing prompt types: {missing_prompts}")

    # Create new DataFrame with only valid data
    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        logging_custom("WARNING", "No final rows after filtering for required prompt types.")
        return final_df

    # --------------------------------------------------------------------
    # Sort, reorder columns, and return
    # --------------------------------------------------------------------
    final_df.sort_values('model_name', inplace=True)

    # Reindex the DataFrame to match the EXACT desired columns (filling missing with None/NaN)
    for col in OUTPUT_FILE_DESIRED_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = None
    final_df = final_df[OUTPUT_FILE_DESIRED_COLUMNS]

    return final_df


# --------------------------------------------------------------------
# 6. Main Execution
# --------------------------------------------------------------------
def main():
    logging_custom("INFO", "Starting report aggregation...")
    df = aggregate_reports()

    if df.empty:
        logging_custom("WARNING", "No valid reports were found or no final data after filtering.")
        logging_custom("WARNING", "No data to save or plot. Exiting.")
        return

    # Save the DataFrame as CSV (the user wants a CSV even though the variable name is .json)
    output_csv = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME.replace(".json", ".csv"))
    df.to_csv(output_csv, index=False)
    logging_custom("INFO", f"Aggregated {len(df)} rows saved to: {output_csv}")

    # If needed, create visualizations or other analysis here
    logging_custom("INFO", "All done!")

if __name__ == "__main__":
    main()
