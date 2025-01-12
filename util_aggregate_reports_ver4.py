#!/usr/bin/env python3
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import logging
from datetime import datetime

# --------------------------------------------------------------------
# 1. Setup logging with a single function to handle different levels
# --------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"log_aggregate_reports_{timestamp}.txt"

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all logging levels

# File handler (DEBUG level messages go to file)
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)

# Console handler (INFO level messages go to console)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter(
    '[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers to the logger
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
        # Default fallback to debug
        logger.debug(f"[INVALID LEVEL: {level}] {message}")


# --------------------------------------------------------------------
# 2. Global constants
# --------------------------------------------------------------------
EVALUATION_RESULTS_DIR = 'evaluation_results_long_20250110'
ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)
OUTPUT_DIR = os.path.join('data', 'aggregate_reports', os.path.basename(EVALUATION_RESULTS_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------
# 3. Helper Functions
# --------------------------------------------------------------------

def extract_model_metadata(model_dir_name):
    """
    Extract model size and quantization information from directory name.
    Example: 'aya-expanse_8b-q4_k_m' => param='8', quant='q4_k_m'
    """
    logging_custom("DEBUG", f"Extracting metadata from: {model_dir_name}")

    param_match = re.search(r'(\d+)b', model_dir_name.lower())
    quant_match = re.search(r'(q4_k_m|fp16)', model_dir_name.lower())

    params = param_match.group(1) if param_match else "unknown"
    quant = quant_match.group(1) if quant_match else "unknown"

    logging_custom("DEBUG", f" => model_params_b={params}, quantization={quant}")
    return {
        'model_params_b': params,
        'quantization': quant
    }

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

    # Flatten some top-level JSON fields
    combined['model_name'] = jdata.get('model_name')
    combined['prompt_type'] = jdata.get('prompt_type')
    combined['total_attempts'] = jdata.get('total_attempts')
    combined['successful_attempts'] = jdata.get('successful_attempts')
    combined['failed_attempts'] = jdata.get('failed_attempts')
    combined['timeout_attempts'] = jdata.get('timeout_attempts')
    combined['avg_execution_time'] = jdata.get('avg_execution_time')
    combined['median_execution_time'] = jdata.get('median_execution_time')
    combined['sd_execution_time'] = jdata.get('sd_execution_time')
    combined['prediction_accuracy'] = jdata.get('prediction_accuracy')
    combined['auc_roc'] = jdata.get('auc_roc')

    # Metadata Averages
    meta_data_averages = jdata.get('meta_data_averages', {})
    combined['avg_total_duration'] = meta_data_averages.get('total_duration')
    combined['avg_load_duration'] = meta_data_averages.get('load_duration')
    combined['avg_prompt_eval_duration'] = meta_data_averages.get('prompt_eval_duration')
    combined['avg_eval_duration'] = meta_data_averages.get('eval_duration')

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

    # Add your text-based stats
    combined['txt_accuracy'] = tdata.get('txt_accuracy')
    combined['txt_auc_roc'] = tdata.get('txt_auc_roc')

    return combined


# --------------------------------------------------------------------
# 4. Core Aggregation Function
# --------------------------------------------------------------------

def aggregate_reports():
    """
    Crawl the directory tree looking for 'reports' directories,
    then find JSON+TXT report pairs. Return a DataFrame of all data.
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

            # Extract metadata from model directory
            model_meta = extract_model_metadata(model_dir_name)

            # Gather JSON files
            json_files = [f for f in files if f.endswith('.json')]
            logging_custom("DEBUG", f" => Found JSON files: {json_files}")

            for jf in json_files:
                base_name = os.path.splitext(jf)[0]
                tf = base_name + '.txt'
                logging_custom("DEBUG", f"Looking for corresponding TXT file: {tf}")

                if tf in files:
                    json_path = os.path.join(root, jf)
                    txt_path = os.path.join(root, tf)
                    logging_custom("INFO", f"Processing pair => JSON: {jf}, TXT: {tf}")

                    combined_stats = process_report_files(json_path, txt_path)
                    if combined_stats:
                        combined_stats.update(model_meta)
                        combined_stats['model_dir'] = model_dir_name
                        aggregated_rows.append(combined_stats)
                        logging_custom("INFO", f"Added data for {jf}")
                else:
                    logging_custom("WARNING", f"No matching TXT for JSON: {jf} in {root}")

    df = pd.DataFrame(aggregated_rows)
    logging_custom("INFO", f"Total aggregated rows: {len(df)}")
    return df


# --------------------------------------------------------------------
# 5. Main Execution
# --------------------------------------------------------------------

def main():
    logging_custom("INFO", "Starting report aggregation...")
    df = aggregate_reports()

    if df.empty:
        logging_custom("WARNING", "No valid reports were found.")
        logging_custom("WARNING", "No data to save or plot. Exiting.")
        return

    # Save the DataFrame
    output_csv = os.path.join(OUTPUT_DIR, "aggregate_model_reports.csv")
    df.to_csv(output_csv, index=False)
    logging_custom("INFO", f"Aggregated {len(df)} rows saved to: {output_csv}")

    logging_custom("INFO", "All done!")


if __name__ == "__main__":
    main()
