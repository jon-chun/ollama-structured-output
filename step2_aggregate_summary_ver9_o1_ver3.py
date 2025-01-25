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

EVALUATION_RESULTS_DIR = 'evaluation_results_long'
ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)
OUTPUT_DIR = os.path.join('aggregate_reports', os.path.basename(EVALUATION_RESULTS_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# We change the output file name as requested
OUTPUT_FILENAME = f"aggregate_model_reports_o1_ver3.csv"

# We define the exact desired columns
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
    """
    logging_custom("DEBUG", f"extract_model_metadata => directory: {model_dir_name}")

    param_match = re.search(r'(\d+)b', model_dir_name.lower())
    quant_match = re.search(r'(q4_k_m|fp16|q8_0|q4_k_g|q4_k_m)', model_dir_name.lower())

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
    cleaned = re.sub(r'[:.]+', '_', cleaned)        # Replace colons, periods w/ underscores
    cleaned = re.sub(r'[^\w]+', '_', cleaned)       # Replace punctuation/spaces
    cleaned = re.sub(r'_+', '_', cleaned)           # Collapse multiple underscores
    return cleaned.strip('_')

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
    Calculate mean, median, and standard deviation for a list of numeric values.
    Returns a dict with keys: 'mean', 'median', 'sd'.
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
# 4. Robust JSON Parsing
# --------------------------------------------------------------------
def parse_report_json(json_path):
    """
    Parse a JSON file. If well-formed, return the parsed data.
    If malformed, fallback to robust line-by-line regex extraction.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging_custom("ERROR", f"Malformed JSON in {json_path} => {e}")
        logging_custom("INFO", f"Attempting fallback parse with regex key-value extraction.")
        return parse_report_json_fallback(json_path)

def parse_report_json_fallback(json_path):
    """
    Fallback method for malformed JSON files.
    We'll attempt to parse lines like:
        "key_name": "value"
        "key_name": 123.45
    and store them in a dict. Very simplistic approach!
    """
    fallback_data = {}
    # Regex can capture "key": "value" or "key": 123.45
    # We'll handle optional quotes around numeric values, etc.
    # This is not a perfect JSON parser, but can salvage partial data from severely malformed JSON.
    pattern = r'"([^"]+)":\s*"?([^",}\]]+)"?'
    # Explanation:
    #  1) "([^"]+)" => group(1) captures the key inside quotes
    #  2) :\s*"? => the colon, optional spaces, optional leading quote
    #  3) ([^",}\]]+) => group(2) captures up to a comma, curly brace, bracket, or quote
    #  4) "? => optional trailing quote

    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        matches = re.finditer(pattern, line)
        for match in matches:
            key = match.group(1).strip()
            val_str = match.group(2).strip()
            # Remove trailing commas if any
            val_str = val_str.rstrip(',')
            # Attempt numeric parse
            # If it fails, store as string
            try:
                # We handle bool or null by a naive approach:
                # If you want to handle them specifically, do it here
                if val_str.lower() == 'true':
                    fallback_data[key] = True
                elif val_str.lower() == 'false':
                    fallback_data[key] = False
                elif val_str.lower() == 'null':
                    fallback_data[key] = None
                else:
                    val_float = float(val_str)
                    fallback_data[key] = val_float
            except ValueError:
                fallback_data[key] = val_str

    return fallback_data

def parse_report_txt(txt_path):
    """
    Parse a TXT file with summary metrics (example lines: 'Accuracy: 53.00%', etc.).
    Return a dict of parsed data. If something is missing or malformed, we skip it.
    """
    logging_custom("DEBUG", f"Parsing TXT file: {txt_path}")
    summary = {}
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logging_custom("ERROR", f"Error reading TXT: {txt_path} => {e}")
        return summary  # return empty

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("Accuracy:"):
            val = line_stripped.split("Accuracy:")[-1].strip().replace('%', '')
            try:
                summary['txt_accuracy'] = float(val)
            except ValueError:
                summary['txt_accuracy'] = None

        if "AUC-ROC:" in line_stripped:
            val = line_stripped.split("AUC-ROC:")[-1].strip()
            try:
                summary['txt_auc_roc'] = float(val)
            except ValueError:
                summary['txt_auc_roc'] = None

    return summary

# --------------------------------------------------------------------
# 5. Processing Function
# --------------------------------------------------------------------
def process_report_files(json_path, txt_path):
    """
    Process a JSON report and a TXT file, returning a combined dict of stats.
    Uses fallback regex if JSON is malformed. Fills data for durations, etc.
    """
    logging_custom("DEBUG", f"process_report_files => JSON: {json_path}, TXT: {txt_path}")

    jdata = parse_report_json(json_path)  # robust parse
    if not jdata:
        logging_custom("WARNING", f"No usable JSON data from {json_path}")
        return None

    tdata = parse_report_txt(txt_path)    # parse .txt metrics

    combined = {}

    # Try to unify model_name/prompt_type from top-level or fallback
    combined['model_name'] = (
        jdata.get('model_name')
        or jdata.get('evaluation', {}).get('model')
        or jdata.get('meta_data', {}).get('model')
    )
    combined['prompt_type'] = (
        jdata.get('prompt_type')
        or jdata.get('evaluation', {}).get('prompt_type')
    )

    # Attempts
    combined['total_attempts'] = jdata.get('total_attempts')
    combined['successful_attempts'] = jdata.get('successful_attempts')
    combined['failed_attempts'] = jdata.get('failed_attempts')
    combined['timeout_attempts'] = jdata.get('timeout_attempts')

    # Execution time stats
    combined['execution_time_mean'] = jdata.get('avg_execution_time')
    combined['execution_time_median'] = jdata.get('median_execution_time')
    combined['execution_time_sd'] = jdata.get('sd_execution_time')

    # Accuracy metrics
    combined['prediction_accuracy'] = jdata.get('prediction_accuracy')
    combined['auc_roc'] = jdata.get('auc_roc')

    # Text-based stats
    combined['txt_accuracy'] = tdata.get('txt_accuracy')
    combined['txt_auc_roc'] = tdata.get('txt_auc_roc')

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

    # Derived confusion metrics
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    combined['precision'] = precision
    combined['recall'] = recall
    combined['f1_score'] = f1_score

    # Get meta_data from top-level or nested
    # We do it robustly, checking multiple possible structures
    raw_meta = jdata.get('meta_data') or jdata.get('evaluation', {}).get('meta_data')
    if raw_meta is None:
        raw_meta = []
    elif isinstance(raw_meta, dict):
        raw_meta = [raw_meta]

    # We'll accumulate durations and counts from meta_data
    total_durations = []
    load_durations = []
    prompt_eval_durations = []
    eval_durations = []
    python_api_durations = []
    prompt_eval_counts = []
    eval_counts = []

    for entry in raw_meta:
        if not isinstance(entry, dict):
            continue
        total_durations.append(entry.get('total_duration'))
        load_durations.append(entry.get('load_duration'))
        prompt_eval_durations.append(entry.get('prompt_eval_duration'))
        eval_durations.append(entry.get('eval_duration'))
        python_api_durations.append(entry.get('python_api_duration'))
        prompt_eval_counts.append(entry.get('prompt_eval_count'))
        eval_counts.append(entry.get('eval_count'))

    # Calculate stats
    total_stats = calculate_stats(total_durations)
    combined['total_duration_mean'] = total_stats['mean']
    combined['total_duration_median'] = total_stats['median']
    combined['total_duration_sd'] = total_stats['sd']

    load_stats = calculate_stats(load_durations)
    combined['load_duration_mean'] = load_stats['mean']
    combined['load_duration_median'] = load_stats['median']
    combined['load_duration_sd'] = load_stats['sd']

    prompt_eval_stats = calculate_stats(prompt_eval_durations)
    combined['prompt_eval_duration_mean'] = prompt_eval_stats['mean']
    combined['prompt_eval_duration_median'] = prompt_eval_stats['median']
    combined['prompt_eval_duration_sd'] = prompt_eval_stats['sd']

    eval_stats = calculate_stats(eval_durations)
    combined['eval_duration_mean'] = eval_stats['mean']
    combined['eval_duration_median'] = eval_stats['median']
    combined['eval_duration_sd'] = eval_stats['sd']

    python_api_stats = calculate_stats(python_api_durations)
    # We do not have columns for python_api_* in the final CSV,
    # but we can handle the missing_count. Just keep it if needed.

    # Prompt eval counts
    prompt_eval_count_stats = calculate_stats(prompt_eval_counts)
    combined['prompt_eval_count_mean'] = prompt_eval_count_stats['mean']
    combined['prompt_eval_count_median'] = prompt_eval_count_stats['median']
    combined['prompt_eval_count_sd'] = prompt_eval_count_stats['sd']

    eval_count_stats = calculate_stats(eval_counts)
    combined['eval_count_mean'] = eval_count_stats['mean']
    combined['eval_count_median'] = eval_count_stats['median']
    combined['eval_count_sd'] = eval_count_stats['sd']

    # Missing count columns
    combined['total_duration_sec_missing_count'] = (1 if not total_durations or total_durations[0] is None else 0)
    combined['load_duration_sec_missing_count'] = (1 if not load_durations or load_durations[0] is None else 0)
    combined['prompt_eval_duration_sec_missing_count'] = (1 if not prompt_eval_durations or prompt_eval_durations[0] is None else 0)
    combined['eval_duration_sec_missing_count'] = (1 if not eval_durations or eval_durations[0] is None else 0)
    combined['python_api_duration_sec_missing_count'] = (1 if not python_api_durations or python_api_durations[0] is None else 0)

    # decision confidence
    confidence_val = None
    decision_dict = jdata.get('decision', {})
    if isinstance(decision_dict, dict):
        confidence_val = decision_dict.get('confidence', None)
    combined['confidence_txt_missing_count'] = (1 if confidence_val is None else 0)

    # prompt_eval_count_missing_count
    combined['prompt_eval_count_missing_count'] = (1 if not prompt_eval_counts or prompt_eval_counts[0] is None else 0)
    # eval_count_missing_count
    combined['eval_count_missing_count'] = (1 if not eval_counts or eval_counts[0] is None else 0)

    return combined


# --------------------------------------------------------------------
# 6. Aggregation and Writing CSV
# --------------------------------------------------------------------
def aggregate_reports():
    """
    Walk the directory tree, find 'reports' folders, gather JSON+TXT pairs.
    Filter to required prompt types, create final DataFrame with all columns.
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

        if os.path.basename(root) == 'reports':
            logging_custom("INFO", f"Found a 'reports' directory: {root}")

            # The parent folder is presumably the model directory
            model_dir_name = os.path.basename(os.path.dirname(root))
            logging_custom("DEBUG", f" => model_dir_name: {model_dir_name}")
            model_meta = {'model_dir': model_dir_name}
            model_meta.update(extract_model_metadata(model_dir_name))

            json_files = [f for f in files if f.endswith('.json')]
            logging_custom("DEBUG", f" => Found JSON files: {json_files}")

            for jf in json_files:
                tf = get_txt_for_json_filename(jf)
                if tf in files:
                    json_path = os.path.join(root, jf)
                    txt_path = os.path.join(root, tf)
                    logging_custom("INFO", f"Processing pair => JSON: {jf}, TXT: {tf}")

                    combined_stats = process_report_files(json_path, txt_path)
                    if combined_stats:
                        combined_stats.update(model_meta)
                        aggregated_rows.append(combined_stats)
                else:
                    logging_custom("WARNING", f"No matching TXT for JSON: {jf} in {root}")

    df = pd.DataFrame(aggregated_rows)
    if df.empty:
        logging_custom("WARNING", "No data extracted. Returning empty DataFrame.")
        return df

    # Filter for required prompt types
    required_prompts = {'PromptType.COT', 'PromptType.COT_NSHOT', 'PromptType.SYSTEM1'}
    final_rows = []
    model_names = df['model_name'].dropna().unique()

    for model in model_names:
        model_df = df[df['model_name'] == model]
        model_prompts = set(model_df['prompt_type'].dropna().unique())
        missing = required_prompts - model_prompts
        if not missing:
            # For each prompt type, pick the first row we find
            for rp in required_prompts:
                subset = model_df[model_df['prompt_type'] == rp]
                if not subset.empty:
                    final_rows.append(subset.iloc[0].to_dict())
            logging_custom("INFO", f"Model {model} has all required prompts.")
        else:
            logging_custom("INFO", f"Skipping model {model}, missing prompts: {missing}")

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        logging_custom("WARNING", "No final rows after prompt-type filtering.")
        return final_df

    final_df.sort_values('model_name', inplace=True)

    # Reindex to ensure all desired columns exist in the final CSV
    for col in OUTPUT_FILE_DESIRED_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = None
    final_df = final_df[OUTPUT_FILE_DESIRED_COLUMNS]

    return final_df

def main():
    logging_custom("INFO", "Starting report aggregation with fallback JSON parse...")
    df = aggregate_reports()
    if df.empty:
        logging_custom("WARNING", "No valid reports found or final data empty.")
        return

    output_csv = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(output_csv, index=False)
    logging_custom("INFO", f"Aggregated {len(df)} rows saved to: {output_csv}")
    logging_custom("INFO", "Done!")

if __name__ == "__main__":
    main()
