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
    quant_match = re.search(r'(q4_k_m|fp16|q4_k_m)', model_dir_name.lower())

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
    combined['execution_time_mean'] = jdata.get('avg_execution_time')
    combined['execution_time_median'] = jdata.get('median_execution_time')
    combined['execution_time_sd'] = jdata.get('sd_execution_time')
    combined['prediction_accuracy'] = jdata.get('prediction_accuracy')
    combined['auc_roc'] = jdata.get('auc_roc')

    # Get the raw duration data arrays from metadata
    meta_data = jdata.get('meta_data', [])
    
    # Initialize arrays for each duration type
    total_durations = []
    load_durations = []
    prompt_eval_durations = []
    eval_durations = []
    
    # Extract duration values from meta_data
    for entry in meta_data:
        if isinstance(entry, dict):
            total_durations.append(entry.get('total_duration'))
            load_durations.append(entry.get('load_duration'))
            prompt_eval_durations.append(entry.get('prompt_eval_duration'))
            eval_durations.append(entry.get('eval_duration'))
    
    # Calculate statistics for each duration type
    def calculate_stats(data):
        """Helper function to calculate mean, median, and standard deviation"""
        clean_data = [x for x in data if x is not None]
        if clean_data:
            return {
                'mean': np.mean(clean_data),
                'median': np.median(clean_data),
                'sd': np.std(clean_data) if len(clean_data) > 1 else 0
            }
        return {'mean': None, 'median': None, 'sd': None}
    
    # Calculate all duration statistics
    total_stats = calculate_stats(total_durations)
    load_stats = calculate_stats(load_durations)
    prompt_eval_stats = calculate_stats(prompt_eval_durations)
    eval_stats = calculate_stats(eval_durations)
    
    # Add all duration statistics to combined dictionary
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

    # Confusion Matrix processing remains the same
    confusion = jdata.get('confusion_matrix', {})
    tp = confusion.get('tp', 0)
    tn = confusion.get('tn', 0)
    fp = confusion.get('fp', 0)
    fn = confusion.get('fn', 0)
    combined['true_positives'] = tp
    combined['true_negatives'] = tn
    combined['false_positives'] = fp
    combined['false_negatives'] = fn

    # Derived metrics remain the same
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
    Now includes filtering for complete prompt type sets and duplicate removal.
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
        
    # Step 1: Group by model_name and process each group
    valid_models_data = []
    
    # Get unique model names
    model_names = df['model_name'].unique()
    
    # Required prompt types
    required_prompts = {'PromptType.COT', 'PromptType.COT_NSHOT', 'PromptType.SYSTEM1'}
    
    for model in model_names:
        # Get all rows for this model
        model_df = df[df['model_name'] == model]
        
        # Get unique prompt types for this model
        model_prompts = set(model_df['prompt_type'].unique())
        
        # Check if model has all required prompt types
        if model_prompts >= required_prompts:
            # Model has all required prompts
            model_rows = []
            
            # For each prompt type, take the first occurrence
            for prompt_type in required_prompts:
                prompt_rows = model_df[model_df['prompt_type'] == prompt_type]
                if not prompt_rows.empty:
                    model_rows.append(prompt_rows.iloc[0])
            
            # If we got exactly three rows (one for each prompt type)
            if len(model_rows) == 3:
                valid_models_data.extend(model_rows)
                logging_custom("INFO", f"Added complete set of prompts for model: {model}")
        else:
            logging_custom("INFO", f"Skipping model {model} - missing prompt types: {required_prompts - model_prompts}")
    
    # Create new DataFrame with only valid data
    final_df = pd.DataFrame(valid_models_data)
    
    logging_custom("INFO", f"Final number of models: {len(final_df['model_name'].unique())}")
    logging_custom("INFO", f"Final number of rows: {len(final_df)}")
    
    return final_df


# --------------------------------------------------------------------
# 6. Main Execution
# --------------------------------------------------------------------
def main():
    logging_custom("INFO", "Starting report aggregation...")
    df = aggregate_reports()

    if df.empty:
        logging_custom("WARNING", "No valid reports were found.")
        logging_custom("WARNING", "No data to save or plot. Exiting.")
        return

    # Save the DataFrame
    output_csv = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(output_csv, index=False)
    logging_custom("INFO", f"Aggregated {len(df)} rows saved to: {output_csv}")

    # If needed, create visualizations or other analysis here
    logging_custom("INFO", "All done!")

if __name__ == "__main__":
    main()