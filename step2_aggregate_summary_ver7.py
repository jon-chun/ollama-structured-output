#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime

# --------------------------------------------------------------------
# 1. Global Constants
# --------------------------------------------------------------------
EVALUATION_RESULTS_DIR = "evaluation_results_short_20250108"
ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)

OUTPUT_DIR = os.path.join("aggregate_reports", os.path.basename(EVALUATION_RESULTS_DIR))
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
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)

# Console handler (INFO-level messages to console)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Common formatter
formatter = logging.Formatter(
    "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
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
    if level == "DEBUG":
        logger.debug(message)
    elif level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "CRITICAL":
        logger.critical(message)
    else:
        logger.debug(f"[INVALID LEVEL: {level}] {message}")


# --------------------------------------------------------------------
# 3. Utility Functions
# --------------------------------------------------------------------
def extract_model_metadata(model_dir_name):
    """
    Extract model size and quantization info from directory name.
    Returns a dict: { 'model_params': ..., 'model_quantization': ... }
    """
    logging_custom("DEBUG", f"extract_model_metadata => directory: {model_dir_name}")

    param_match = re.search(r"(\d+)b", model_dir_name.lower())
    quant_match = re.search(r"(q4_k_m|fp16)", model_dir_name.lower())

    params = param_match.group(1) if param_match else "unknown"
    quant = quant_match.group(1) if quant_match else "unknown"

    logging_custom("DEBUG", f" -> model_params={params}, model_quantization={quant}")
    return {
        "model_params": params,
        "model_quantization": quant,
    }

def get_txt_for_json_filename(json_filename: str) -> str:
    """
    Given a JSON filename, derive the matching TXT filename.
    """
    base_name, _ = os.path.splitext(json_filename)
    if base_name.startswith("metrics_"):
        txt_name = "report_" + base_name[len("metrics_"):]
    else:
        txt_name = base_name
    return txt_name + ".txt"

# --- Fallback JSON parsing to handle slight malformation -----------------
def try_parse_json(content):
    """
    Attempt standard json.loads first.
    If that fails, try removing trailing commas from the content and parse again.
    If all attempts fail, return None.
    """
    # 1) direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging_custom("DEBUG", f"Standard json.loads failed => {e}")

    # 2) remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r",\s*([\]}])", r"\1", content)
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        logging_custom("DEBUG", f"Fallback removing trailing commas failed => {e}")
        return None

def parse_report_json(json_path):
    """
    Open a JSON file, attempt multiple fallback parse strategies.
    Return the loaded object or None if everything fails.
    """
    logging_custom("DEBUG", f"Parsing JSON file: {json_path}")

    if not os.path.exists(json_path):
        logging_custom("ERROR", f"JSON file missing: {json_path}")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()
        data = try_parse_json(content)
        if data is None:
            logging_custom("ERROR", f"All JSON parse attempts failed for: {json_path}")
        return data
    except Exception as e:
        logging_custom("ERROR", f"Error reading JSON: {json_path} => {e}")
        return None

def parse_report_txt(txt_path):
    """
    Parse a TXT file with summary metrics (example line: "Accuracy: 53.00%").
    Return a dict, e.g. { 'txt_accuracy': 53.0, 'txt_auc_roc': 0.47 }
    """
    logging_custom("DEBUG", f"Parsing TXT file: {txt_path}")
    summary = {}

    if not os.path.exists(txt_path):
        logging_custom("WARNING", f"TXT file missing: {txt_path}")
        return summary

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logging_custom("ERROR", f"Error reading TXT: {txt_path} => {e}")
        return summary

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("Accuracy:"):
            val = line_stripped.split("Accuracy:")[-1].strip().replace("%", "")
            try:
                summary["txt_accuracy"] = float(val)
            except ValueError:
                summary["txt_accuracy"] = None

        if "AUC-ROC:" in line_stripped:
            val = line_stripped.split("AUC-ROC:")[-1].strip()
            try:
                summary["txt_auc_roc"] = float(val)
            except ValueError:
                summary["txt_auc_roc"] = None

    logging_custom("DEBUG", f"Parsed TXT summary => {summary}")
    return summary


# --------------------------------------------------------------------
# 4. Gathering Raw Data
# --------------------------------------------------------------------

# A dictionary of synonyms (if your JSON uses different keys).
# For example, if your JSON has "evaluation_time_mean" but code wants "execution_time_mean".
FIELD_SYNONYMS = {
    # aggregator expects => possible JSON synonyms
    "execution_time_mean": ["evaluation_time_mean", "exec_time_mean"],
    "execution_time_median": ["evaluation_time_median", "exec_time_median"],
    "execution_time_sd": ["evaluation_time_sd", "exec_time_sd"],
    "total_duration": ["duration_total", "eval_time_total"],
    "load_duration": ["duration_load", "load_time"],
    "prompt_eval_duration": ["prompt_eval_time"],
    "eval_duration": ["evaluation_time", "eval_time"],
    # etc.  Expand if you find more mismatches
}

def apply_synonyms(jdata, key, default=None):
    """
    Looks up `key` in jdata.
    If not found, tries synonyms from FIELD_SYNONYMS.
    Returns the first match found or `default`.
    """
    # 1) direct check
    if key in jdata:
        return jdata[key]

    # 2) check synonyms
    if key in FIELD_SYNONYMS:
        for alt in FIELD_SYNONYMS[key]:
            if alt in jdata:
                logging_custom("DEBUG", f"Found synonym '{alt}' for '{key}' in jdata.")
                return jdata[alt]

    return default


def process_report_files(json_path, txt_path, model_dir_name):
    """
    Parse the JSON + TXT for one "reports" file pair.
    Return a *list of raw rows* (dicts).
    Each row represents a single meta_data record (or 'call').
    """
    logging_custom("INFO", f"process_report_files => JSON: {json_path}, TXT: {txt_path}")

    jdata = parse_report_json(json_path)
    if not jdata:
        logging_custom("ERROR", f"Failed to parse JSON: {json_path}, skipping.")
        return []

    # Print top-level keys to see what's in the JSON
    if isinstance(jdata, dict):
        logging_custom("DEBUG", f"Top-level JSON keys => {list(jdata.keys())}")

    tdata = parse_report_txt(txt_path)

    # Identify model_dir, model_params, model_quantization
    model_meta = extract_model_metadata(model_dir_name)

    # We'll build a list of "raw" rows
    out_rows = []

    # Try to infer model_name, prompt_type from jdata
    model_name = None
    prompt_type = None

    eval_block = jdata.get("evaluation", {})
    if isinstance(eval_block, dict):
        model_name = eval_block.get("model")
        prompt_type = eval_block.get("prompt_type")

    if not model_name:
        meta_blk = jdata.get("meta_data", {})
        if isinstance(meta_blk, dict):
            model_name = meta_blk.get("model")

    if not model_name:
        model_name = jdata.get("model_name", "unknown")

    if not prompt_type:
        prompt_type = jdata.get("prompt_type", "PromptType.UNKNOWN")

    logging_custom("DEBUG", f" => model_name={model_name}, prompt_type={prompt_type}")

    # Gather top-level fields
    # We'll apply synonyms if needed
    confusion = jdata.get("confusion_matrix", {})
    top_level = {
        "model_name": model_name,
        "prompt_type": prompt_type,

        "total_attempts": jdata.get("total_attempts"),
        "successful_attempts": jdata.get("successful_attempts"),
        "failed_attempts": jdata.get("failed_attempts"),
        "timeout_attempts": jdata.get("timeout_attempts"),

        "execution_time_mean": apply_synonyms(jdata, "execution_time_mean"),
        "execution_time_median": apply_synonyms(jdata, "execution_time_median"),
        "execution_time_sd": apply_synonyms(jdata, "execution_time_sd"),
        "prediction_accuracy": jdata.get("prediction_accuracy"),
        "auc_roc": jdata.get("auc_roc"),

        "true_positives": confusion.get("tp", 0),
        "true_negatives": confusion.get("tn", 0),
        "false_positives": confusion.get("fp", 0),
        "false_negatives": confusion.get("fn", 0),

        "precision": None,
        "recall": None,
        "f1_score": None,

        # from the TXT summary
        "txt_accuracy": tdata.get("txt_accuracy"),
        "txt_auc_roc": tdata.get("txt_auc_roc"),

        # from the model_dir metadata
        "model_dir": model_dir_name,
        "model_params": model_meta["model_params"],
        "model_quantization": model_meta["model_quantization"],
    }

    # Next, handle meta_data. Could be dict, list, or absent.
    meta_data = jdata.get("meta_data")
    if meta_data:
        logging_custom("DEBUG", f"meta_data type => {type(meta_data)}")
    else:
        logging_custom("DEBUG", f"No meta_data present in {json_path}")

    # If there's no meta_data, we add one row with None for durations
    if not meta_data:
        row = dict(top_level)
        row["total_duration"] = None
        row["load_duration"] = None
        row["prompt_eval_duration"] = None
        row["eval_duration"] = None
        row["prompt_eval_count"] = None
        row["eval_count"] = None
        logging_custom("DEBUG", f"Adding row with no meta_data => {row}")
        out_rows.append(row)
        return out_rows

    # unify single dict vs list
    if isinstance(meta_data, dict):
        meta_data_list = [meta_data]
    elif isinstance(meta_data, list):
        meta_data_list = meta_data
    else:
        logging_custom("DEBUG", f"meta_data is neither dict nor list => {type(meta_data)}. Skipping.")
        return []

    for record in meta_data_list:
        if not isinstance(record, dict):
            logging_custom("DEBUG", f"Skipping non-dict record => {record}")
            continue

        # Print sub-keys so we can see if it has total_duration, etc.
        logging_custom("DEBUG", f"meta_data record keys => {list(record.keys())}")

        row = dict(top_level)

        # Here we do synonyms for each field we expect in meta_data
        row["total_duration"] = apply_synonyms(record, "total_duration")
        row["load_duration"] = apply_synonyms(record, "load_duration")
        row["prompt_eval_duration"] = apply_synonyms(record, "prompt_eval_duration")
        row["eval_duration"] = apply_synonyms(record, "eval_duration")

        # Some older code had these as well
        row["prompt_eval_count"] = record.get("prompt_eval_count")
        row["eval_count"] = record.get("eval_count")

        logging_custom("DEBUG", f"Raw row => {row}")
        out_rows.append(row)

    return out_rows


# --------------------------------------------------------------------
# 5. Core Aggregation
# --------------------------------------------------------------------
def aggregate_reports():
    """
    1) Crawl the directory for 'reports' folders.
    2) For each JSON+TXT pair, accumulate raw data rows.
    3) Group by (model_name, prompt_type) -> produce final aggregated stats.
    4) Return final DataFrame.
    """
    logging_custom("INFO", f"Starting directory crawl from: {ROOT_DIR}")

    if not os.path.exists(ROOT_DIR):
        logging_custom("ERROR", f"Evaluation directory not found: {ROOT_DIR}")
        return pd.DataFrame()

    all_raw_rows = []

    for root, dirs, files in os.walk(ROOT_DIR):
        logging_custom("DEBUG", f"Visiting directory => {root}")
        if os.path.basename(root) == "reports":
            logging_custom("INFO", f"Found a 'reports' folder => {root}")
            model_dir_name = os.path.basename(os.path.dirname(root))
            logging_custom("DEBUG", f"model_dir_name => {model_dir_name}")

            json_files = [f for f in files if f.endswith(".json")]
            for jf in json_files:
                tf = get_txt_for_json_filename(jf)
                json_path = os.path.join(root, jf)
                txt_path = os.path.join(root, tf)
                logging_custom("DEBUG", f"Checking pair => JSON={jf}, TXT={tf}")

                if os.path.exists(txt_path):
                    raw_rows = process_report_files(json_path, txt_path, model_dir_name)
                    logging_custom("DEBUG", f" -> raw_rows count={len(raw_rows)} from {jf}")
                    all_raw_rows.extend(raw_rows)
                else:
                    logging_custom("WARNING", f"No matching TXT for JSON: {jf} in {root}")

    df_raw = pd.DataFrame(all_raw_rows)
    logging_custom("INFO", f"Constructed df_raw with shape => {df_raw.shape}")
    if not df_raw.empty:
        logging_custom("INFO", f"df_raw columns => {list(df_raw.columns)}")
        logging_custom("DEBUG", f"df_raw sample =>\n{df_raw.head(5)}")

    if df_raw.empty:
        logging_custom("WARNING", "No raw data found; returning empty DataFrame.")
        return df_raw

    numeric_cols = [
        "total_attempts", "successful_attempts", "failed_attempts", "timeout_attempts",
        "execution_time_mean", "execution_time_median", "execution_time_sd",
        "prediction_accuracy", "auc_roc",
        "true_positives", "true_negatives", "false_positives", "false_negatives",
        "txt_accuracy", "txt_auc_roc",
        "total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
        "prompt_eval_count", "eval_count"
    ]
    for col in numeric_cols:
        if col in df_raw.columns:
            logging_custom("DEBUG", f"Converting column '{col}' to numeric.")
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    logging_custom("DEBUG", f"After numeric conversion, df_raw dtypes =>\n{df_raw.dtypes}")

    # Optional: Check if any key columns are entirely NaN
    # This will alert us that the aggregator won't produce anything
    for col in ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration"]:
        if col in df_raw.columns:
            all_nan = df_raw[col].isna().all()
            if all_nan:
                logging_custom("WARNING", f"Column {col} is ALL NaN. No data for durations in that field!")

    group_cols = ["model_name", "prompt_type"]
    agg_dict = {
        "total_attempts": "sum",
        "successful_attempts": "sum",
        "failed_attempts": "sum",
        "timeout_attempts": "sum",

        "execution_time_mean": "mean",
        "execution_time_median": "mean",
        "execution_time_sd": "mean",
        "prediction_accuracy": "mean",
        "auc_roc": "mean",

        "total_duration": ["mean", "median", "std"],
        "load_duration": ["mean", "median", "std"],
        "prompt_eval_duration": ["mean", "median", "std"],
        "eval_duration": ["mean", "median", "std"],

        "true_positives": "sum",
        "true_negatives": "sum",
        "false_positives": "sum",
        "false_negatives": "sum",

        "txt_accuracy": "mean",
        "txt_auc_roc": "mean",

        "model_dir": "first",
        "model_params": "first",
        "model_quantization": "first",
    }

    logging_custom("DEBUG", f"Grouping by => {group_cols}, aggregator => {list(agg_dict.keys())}")
    df_agg = df_raw.groupby(group_cols, as_index=False).agg(agg_dict)

    logging_custom("DEBUG", f"df_agg shape => {df_agg.shape}, columns => {df_agg.columns.tolist()}")
    if not df_agg.empty:
        logging_custom("DEBUG", f"df_agg sample =>\n{df_agg.head(5)}")

    # Flatten multi-index
    new_cols = []
    for col in df_agg.columns:
        if isinstance(col, tuple):
            base_name, agg_name = col
            if agg_name in ["mean", "median", "std"]:
                new_col = f"{base_name}_{agg_name}"
            else:
                new_col = base_name
            new_cols.append(new_col)
        else:
            new_cols.append(col)
    df_agg.columns = new_cols

    # rename _std => _sd
    for c in df_agg.columns:
        if c.endswith("_std"):
            df_agg.rename(columns={c: c.replace("_std", "_sd")}, inplace=True)

    # Recompute precision, recall, f1
    def safe_div(n, d):
        return n / d if d != 0 else 0.0

    df_agg["precision"] = df_agg.apply(
        lambda r: safe_div(r["true_positives"], (r["true_positives"] + r["false_positives"])),
        axis=1
    )
    df_agg["recall"] = df_agg.apply(
        lambda r: safe_div(r["true_positives"], (r["true_positives"] + r["false_negatives"])),
        axis=1
    )
    df_agg["f1_score"] = df_agg.apply(
        lambda r: safe_div(2.0 * r["precision"] * r["recall"], (r["precision"] + r["recall"])),
        axis=1
    )

    # Final column order
    desired_cols = [
        "model_name","prompt_type",
        "total_attempts","successful_attempts","failed_attempts","timeout_attempts",
        "execution_time_mean","execution_time_median","execution_time_sd",
        "prediction_accuracy","auc_roc",
        "total_duration_mean","total_duration_median","total_duration_sd",
        "load_duration_mean","load_duration_median","load_duration_sd",
        "prompt_eval_duration_mean","prompt_eval_duration_median","prompt_eval_duration_sd",
        "eval_duration_mean","eval_duration_median","eval_duration_sd",
        "true_positives","true_negatives","false_positives","false_negatives",
        "precision","recall","f1_score",
        "txt_accuracy","txt_auc_roc",
        "model_dir","model_params","model_quantization"
    ]

    # Ensure columns exist
    for c in desired_cols:
        if c not in df_agg.columns:
            df_agg[c] = None

    df_agg = df_agg[desired_cols]

    # Check final aggregator columns
    logging_custom("DEBUG", f"FINAL df_agg shape => {df_agg.shape}")
    logging_custom("DEBUG", f"FINAL df_agg columns => {df_agg.columns.tolist()}")
    if not df_agg.empty:
        logging_custom("DEBUG", f"FINAL df_agg sample =>\n{df_agg.head(5)}")

    # Debug: see if total_duration_mean or load_duration_mean are all NaN:
    for col in ["total_duration_mean", "load_duration_mean", "prompt_eval_duration_mean", "eval_duration_mean"]:
        if col in df_agg.columns:
            all_nan = df_agg[col].isna().all()
            if all_nan:
                logging_custom("WARNING", f"[AGG] Column {col} is ALL NaN (no data after grouping).")

    return df_agg


# --------------------------------------------------------------------
# 6. Main Execution
# --------------------------------------------------------------------
def main():
    logging_custom("INFO", "Starting report aggregation...")
    df_final = aggregate_reports()

    if df_final.empty:
        logging_custom("WARNING", "No valid data was found. Exiting.")
        return

    output_csv = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df_final.to_csv(output_csv, index=False)
    logging_custom("INFO", f"Final aggregated data saved to: {output_csv}")
    logging_custom("INFO", "All done!")


if __name__ == "__main__":
    main()
