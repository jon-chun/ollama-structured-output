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

    # Example match "72b", "8b", etc.
    param_match = re.search(r"(\d+)b", model_dir_name.lower())
    # Example match "q4_k_m" or "fp16"
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
    If JSON is prefixed with 'metrics_', then the TXT is 'report_*'.
    Otherwise, the same base name.
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
        pass

    # 2) remove trailing commas at end of lines or objects
    #    (very naive approach, but might fix some minor issues)
    try:
        # remove trailing commas before closing braces/brackets
        # e.g. '},' => '}'
        fixed = re.sub(r",\s*([\]}])", r"\1", content)
        return json.loads(fixed)
    except json.JSONDecodeError as e:
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

    return summary


# --------------------------------------------------------------------
# 4. Gathering Raw Data
# --------------------------------------------------------------------
def process_report_files(json_path, txt_path, model_dir_name):
    """
    Parse the JSON + TXT for one "reports" file pair.
    Return a *list of raw rows* (dicts).
    Each row represents a single meta_data record (or 'call').
    """
    logging_custom("DEBUG", f"process_report_files => JSON: {json_path}, TXT: {txt_path}")

    # Try to load JSON
    jdata = parse_report_json(json_path)
    if not jdata:
        # Could not parse => skip
        logging_custom("ERROR", f"Failed to parse JSON: {json_path}")
        return []

    # Try to load TXT
    tdata = parse_report_txt(txt_path)

    # Identify model_dir, model_params, model_quantization
    model_meta = extract_model_metadata(model_dir_name)

    # We'll build a list of "raw" rows
    out_rows = []

    # We can pick either "evaluation" or "meta_data" to get model_name/prompt_type.
    # Some JSON might store them differently, so let's do fallback:
    model_name = None
    prompt_type = None

    # 1) Attempt jdata["evaluation"]
    eval_block = jdata.get("evaluation", {})
    if isinstance(eval_block, dict):
        model_name = eval_block.get("model")
        prompt_type = eval_block.get("prompt_type")

    # 2) If still missing, attempt jdata["meta_data"] => "model"
    if not model_name:
        meta_blk = jdata.get("meta_data", {})
        if isinstance(meta_blk, dict):
            model_name = meta_blk.get("model")

    # 3) If still missing, fallback to jdata["model_name"] or "unknown"
    if not model_name:
        model_name = jdata.get("model_name", "unknown")

    if not prompt_type:
        prompt_type = jdata.get("prompt_type", "PromptType.UNKNOWN")

    # Gather top-level fields
    # (some might be absent, we'll just get None)
    top_level = {
        "model_name": model_name,
        "prompt_type": prompt_type,

        "total_attempts": jdata.get("total_attempts"),
        "successful_attempts": jdata.get("successful_attempts"),
        "failed_attempts": jdata.get("failed_attempts"),
        "timeout_attempts": jdata.get("timeout_attempts"),

        "execution_time_mean": jdata.get("execution_time_mean"),
        "execution_time_median": jdata.get("execution_time_median"),
        "execution_time_sd": jdata.get("execution_time_sd"),
        "prediction_accuracy": jdata.get("prediction_accuracy"),
        "auc_roc": jdata.get("auc_roc"),

        # confusion matrix
        "true_positives": jdata.get("confusion_matrix", {}).get("tp", 0),
        "true_negatives": jdata.get("confusion_matrix", {}).get("tn", 0),
        "false_positives": jdata.get("confusion_matrix", {}).get("fp", 0),
        "false_negatives": jdata.get("confusion_matrix", {}).get("fn", 0),

        # We'll recalc precision/recall/f1 after grouping, so store placeholders
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

    # Now handle meta_data. Could be dict, list, or absent.
    meta_data = jdata.get("meta_data")
    if not meta_data:
        # produce at least one row with None durations
        row = dict(top_level)
        row["total_duration"] = None
        row["load_duration"] = None
        row["prompt_eval_duration"] = None
        row["eval_duration"] = None
        row["prompt_eval_count"] = None
        row["eval_count"] = None
        out_rows.append(row)
        return out_rows

    # unify single dict vs list
    if isinstance(meta_data, dict):
        meta_data_list = [meta_data]
    elif isinstance(meta_data, list):
        meta_data_list = meta_data
    else:
        meta_data_list = []

    for record in meta_data_list:
        if not isinstance(record, dict):
            continue
        row = dict(top_level)
        row["total_duration"] = record.get("total_duration")
        row["load_duration"] = record.get("load_duration")
        row["prompt_eval_duration"] = record.get("prompt_eval_duration")
        row["eval_duration"] = record.get("eval_duration")
        # Also add these if they exist
        row["prompt_eval_count"] = record.get("prompt_eval_count")
        row["eval_count"] = record.get("eval_count")

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

    # Walk the entire tree
    for root, dirs, files in os.walk(ROOT_DIR):
        if os.path.basename(root) == "reports":
            logging_custom("INFO", f"Found reports folder: {root}")
            model_dir_name = os.path.basename(os.path.dirname(root))
            logging_custom("DEBUG", f"model_dir_name => {model_dir_name}")

            json_files = [f for f in files if f.endswith(".json")]
            for jf in json_files:
                tf = get_txt_for_json_filename(jf)
                json_path = os.path.join(root, jf)
                txt_path = os.path.join(root, tf)

                if os.path.exists(txt_path):
                    raw_rows = process_report_files(json_path, txt_path, model_dir_name)
                    if raw_rows:
                        all_raw_rows.extend(raw_rows)
                else:
                    logging_custom("WARNING", f"No matching TXT for JSON: {jf} in {root}")

    # Build DF
    df_raw = pd.DataFrame(all_raw_rows)
    if df_raw.empty:
        logging_custom("WARNING", "No raw data found; returning empty DataFrame.")
        return df_raw

    # Convert relevant columns to numeric
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
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Set up aggregator
    # For durations we do [mean, median, std].
    # For attempt counters and confusion matrix, we sum them.
    # For top-level time stats (execution_time_mean, etc.), let's do "mean".
    group_cols = ["model_name", "prompt_type"]

    agg_dict = {
        # Summation for attempt counters
        "total_attempts": "sum",
        "successful_attempts": "sum",
        "failed_attempts": "sum",
        "timeout_attempts": "sum",

        # Average these top-level time stats
        "execution_time_mean": "mean",
        "execution_time_median": "mean",
        "execution_time_sd": "mean",
        "prediction_accuracy": "mean",
        "auc_roc": "mean",

        # durations
        "total_duration": ["mean", "median", "std"],
        "load_duration": ["mean", "median", "std"],
        "prompt_eval_duration": ["mean", "median", "std"],
        "eval_duration": ["mean", "median", "std"],

        # confusion matrix
        "true_positives": "sum",
        "true_negatives": "sum",
        "false_positives": "sum",
        "false_negatives": "sum",

        # text-based stats => average them
        "txt_accuracy": "mean",
        "txt_auc_roc": "mean",

        # We'll keep the first of these (they should be consistent)
        "model_dir": "first",
        "model_params": "first",
        "model_quantization": "first",
    }

    df_agg = df_raw.groupby(group_cols, as_index=False).agg(agg_dict)

    # Flatten multi-level columns
    new_cols = []
    for col in df_agg.columns:
        if isinstance(col, tuple):
            base_name, agg_name = col
            # e.g. ("total_duration", "mean") => "total_duration_mean"
            if agg_name in ["mean", "median", "std"]:
                new_col = f"{base_name}_{agg_name}"
            else:
                new_col = base_name
            new_cols.append(new_col)
        else:
            new_cols.append(col)
    df_agg.columns = new_cols

    # rename _std to _sd if desired
    for c in df_agg.columns:
        if c.endswith("_std"):
            df_agg.rename(columns={c: c.replace("_std", "_sd")}, inplace=True)

    # Recompute precision/recall/f1 from the summed confusion matrix
    # Avoid division by zero
    df_agg["precision"] = df_agg.apply(
        lambda r: (r["true_positives"] / (r["true_positives"] + r["false_positives"]))
                  if (r["true_positives"] + r["false_positives"]) > 0 else 0.0,
        axis=1
    )
    df_agg["recall"] = df_agg.apply(
        lambda r: (r["true_positives"] / (r["true_positives"] + r["false_negatives"]))
                  if (r["true_positives"] + r["false_negatives"]) > 0 else 0.0,
        axis=1
    )
    df_agg["f1_score"] = df_agg.apply(
        lambda r: (2.0 * r["precision"] * r["recall"] / (r["precision"] + r["recall"]))
                  if (r["precision"] + r["recall"]) > 0 else 0.0,
        axis=1
    )

    # Now define final column ordering per your specs
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

    # Ensure these columns exist; fill with None if missing
    for c in desired_cols:
        if c not in df_agg.columns:
            df_agg[c] = None

    # Reorder columns
    df_agg = df_agg[desired_cols]

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

    # Save the aggregated DataFrame
    output_csv = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df_final.to_csv(output_csv, index=False)
    logging_custom("INFO", f"Final aggregated data saved to: {output_csv}")
    logging_custom("INFO", "All done!")


if __name__ == "__main__":
    main()
