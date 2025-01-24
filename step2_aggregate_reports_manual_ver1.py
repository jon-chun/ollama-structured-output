#!/usr/bin/env python3
"""
step2_aggregate_reports_manual_ver1.py

This script scans a directory of JSON evaluation reports, extracts relevant
metrics from each file, and writes a single aggregated CSV summary.

It is specialized to handle three "prompt type" variants (system1, CoT, CoT-nshot)
for each of a set of known models, but can be easily extended as needed.

Usage:
  python step2_aggregate_reports_manual_ver1.py

Adjust as needed for your local paths, logging, or naming conventions.
"""

import os
import json
import csv
import math
import logging
from typing import Dict, Any

# -----------------------------------------------------------------------------
# 1. Define global variables
# -----------------------------------------------------------------------------
INPUT_ROOT_DIR = os.path.join('evaluation_reports_manual')
OUTPUT_ROOT_DIR = os.path.join('evaluation_reports_summary')

PROMPT_TYPE_LS = ['_cot_', '_cot-nshot_', '_system1_']

# ENSEMBLE_NAME = "test-20250124-0053"
# MODEL_LS = ["athene-v2:72b-q4_K_M", "aya-expanse:8b-q4_K_M", "command-r:35b-08-2024-q4_K_M"]

ENSEMBLE_NAME = "all-standardllm"
MODEL_LS = [
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

# Ensure the output directory exists
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Support functions
# -----------------------------------------------------------------------------

def parse_precision_recall_f1(tp: int, tn: int, fp: int, fn: int) -> (float, float, float):
    """
    Given confusion matrix counts, compute precision, recall and F1 score.
    """
    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    return precision, recall, f1_score

def parse_model_fields(model_name: str) -> (str, str, str):
    """
    Return (model_dir, model_params, model_quantization) from the model_name.
    This is somewhat arbitrary and may need to be adapted for your naming conventions.
    """
    # Example approach: replace ":" with "_" for a directory-like name
    model_dir = model_name.replace(":", "_")

    # A naive attempt to parse out "params" (like "72b", "8b", "35b-08-2024", etc)
    # and "q4" or "qX" as quantization if present.
    # If these patterns are not consistent, adjust this function accordingly.
    chunks = model_name.split(":")
    if len(chunks) == 2:
        # second chunk might look like "72b-q4_K_M"
        after_colon = chunks[1]
        # pick out something like "72b" as model_params
        # and "q4" or "q8" as model_quantization if we can find them.
        # We'll do a quick parse:
        subchunks = after_colon.split("-")
        # By convention, let's take subchunks[0] as model_params
        model_params = subchunks[0] if subchunks else "unknown"
        # Then search for "q4" or "q8" in subchunks
        model_quantization = "unknown"
        for sc in subchunks:
            if sc.startswith("q"):
                model_quantization = sc
                break
    else:
        model_params = "unknown"
        model_quantization = "unknown"

    return model_dir, model_params, model_quantization

def process_system1(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Specialized processing for 'PromptType.SYSTEM1' type files.
    Returns a dict of aggregated metrics suitable for CSV.
    """
    return parse_common_fields(json_data)

def process_cot(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Specialized processing for 'PromptType.COT' type files.
    Returns a dict of aggregated metrics suitable for CSV.
    """
    return parse_common_fields(json_data)

def process_cot_nshot(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Specialized processing for 'PromptType.COT_NSHOT' type files.
    Returns a dict of aggregated metrics suitable for CSV.
    """
    return parse_common_fields(json_data)

def parse_common_fields(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Common field extraction for each JSON file, given they share a common structure.
    Returns a dictionary of metrics to place in CSV row.
    """
    # Basic fields
    model_name = json_data.get("model_name", "")
    prompt_type = json_data.get("prompt_type", "")
    total_attempts = json_data.get("total_attempts", 0)
    successful_attempts = json_data.get("successful_attempts", 0)
    failed_attempts = json_data.get("failed_attempts", 0)
    timeout_attempts = json_data.get("timeout_attempts", 0)

    execution_time_mean = json_data.get("avg_execution_time", 0.0)
    execution_time_median = json_data.get("median_execution_time", 0.0)
    execution_time_sd = json_data.get("sd_execution_time", 0.0)

    prediction_accuracy = json_data.get("prediction_accuracy", 0.0)
    auc_roc = json_data.get("auc_roc", None)  # May be None if not present

    # For CSV output, let's put these textual placeholders if needed
    txt_accuracy = prediction_accuracy if prediction_accuracy else None
    txt_auc_roc = auc_roc if auc_roc else None

    # Extract meta_data_averages
    meta_avg = json_data.get("meta_data_averages", {})
    meta_sd = json_data.get("meta_data_sd", {})

    total_duration_mean = meta_avg.get("total_duration", 0.0)
    total_duration_sd = meta_sd.get("total_duration", 0.0)
    # We do not have a median for total_duration in the sample JSON, so set to None or 0
    total_duration_median = None  # the example CSV might or might not have it

    load_duration_mean = meta_avg.get("load_duration", 0.0)
    load_duration_sd = meta_sd.get("load_duration", 0.0)
    load_duration_median = None

    prompt_eval_duration_mean = meta_avg.get("prompt_eval_duration", 0.0)
    prompt_eval_duration_sd = meta_sd.get("prompt_eval_duration", 0.0)
    prompt_eval_duration_median = None

    eval_duration_mean = meta_avg.get("eval_duration", 0.0)
    eval_duration_sd = meta_sd.get("eval_duration", 0.0)
    eval_duration_median = None

    prompt_eval_count_mean = meta_avg.get("prompt_eval_count", 0.0)
    prompt_eval_count_sd = meta_sd.get("prompt_eval_count", 0.0)
    prompt_eval_count_median = None

    eval_count_mean = meta_avg.get("eval_count", 0.0)
    eval_count_sd = meta_sd.get("eval_count", 0.0)
    eval_count_median = None

    # Confusion matrix
    confusion = json_data.get("confusion_matrix", {})
    tp = confusion.get("tp", 0)
    tn = confusion.get("tn", 0)
    fp = confusion.get("fp", 0)
    fn = confusion.get("fn", 0)

    precision, recall, f1_score = parse_precision_recall_f1(tp, tn, fp, fn)

    # Additional placeholders
    total_duration_sec_missing_count = 0
    load_duration_sec_missing_count = 0
    prompt_eval_duration_sec_missing_count = 0
    eval_duration_sec_missing_count = 0
    python_api_duration_sec_missing_count = 0
    confidence_txt_missing_count = 0
    prompt_eval_count_missing_count = 0
    eval_count_missing_count = 0

    # Parse extra model fields (for CSV columns: model_dir, model_params, model_quantization)
    model_dir, model_params, model_quantization = parse_model_fields(model_name)

    row_data = {
        "model_name": model_name,
        "prompt_type": prompt_type.lower().replace("prompttype.", ""),  # e.g. system1, cot, cot_nshot, ...
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "failed_attempts": failed_attempts,
        "timeout_attempts": timeout_attempts,

        "execution_time_mean": execution_time_mean,
        "execution_time_median": execution_time_median,
        "execution_time_sd": execution_time_sd,

        "prediction_accuracy": prediction_accuracy,
        "auc_roc": auc_roc if auc_roc is not None else "",

        # placeholders for textual versions of accuracy/auc if desired
        "txt_accuracy": txt_accuracy if txt_accuracy is not None else "",
        "txt_auc_roc": txt_auc_roc if txt_auc_roc is not None else "",

        "total_duration_mean": total_duration_mean,
        "total_duration_median": total_duration_median if total_duration_median is not None else "",
        "total_duration_sd": total_duration_sd,

        "load_duration_mean": load_duration_mean,
        "load_duration_median": load_duration_median if load_duration_median is not None else "",
        "load_duration_sd": load_duration_sd,

        "prompt_eval_duration_mean": prompt_eval_duration_mean,
        "prompt_eval_duration_median": prompt_eval_duration_median if prompt_eval_duration_median is not None else "",
        "prompt_eval_duration_sd": prompt_eval_duration_sd,

        "eval_duration_mean": eval_duration_mean,
        "eval_duration_median": eval_duration_median if eval_duration_median is not None else "",
        "eval_duration_sd": eval_duration_sd,

        "prompt_eval_count_mean": prompt_eval_count_mean,
        "prompt_eval_count_median": prompt_eval_count_median if prompt_eval_count_median is not None else "",
        "prompt_eval_count_sd": prompt_eval_count_sd,

        "eval_count_mean": eval_count_mean,
        "eval_count_median": eval_count_median if eval_count_median is not None else "",
        "eval_count_sd": eval_count_sd,

        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,

        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,

        "model_dir": model_dir,
        "model_params": model_params,
        "model_quantization": model_quantization,

        "total_duration_sec_missing_count": total_duration_sec_missing_count,
        "load_duration_sec_missing_count": load_duration_sec_missing_count,
        "prompt_eval_duration_sec_missing_count": prompt_eval_duration_sec_missing_count,
        "eval_duration_sec_missing_count": eval_duration_sec_missing_count,
        "python_api_duration_sec_missing_count": python_api_duration_sec_missing_count,
        "confidence_txt_missing_count": confidence_txt_missing_count,
        "prompt_eval_count_missing_count": prompt_eval_count_missing_count,
        "eval_count_missing_count": eval_count_missing_count
    }
    return row_data

# -----------------------------------------------------------------------------
# 3. Main aggregator
# -----------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    # We'll accumulate rows in a list of dicts
    csv_rows = []

    logging.info(f"Scanning input directory: {INPUT_ROOT_DIR}")
    all_files = os.listdir(INPUT_ROOT_DIR)

    # We'll check for each model and each prompt substring if there's a matching file
    # and parse it. Alternatively, we can just parse every JSON in the directory and
    # detect model/prompt from the content. We'll do a simpler approach:
    #   - If a file name contains any of the prompt_type substrings
    #     and also contains the model name, parse it.

    for fname in all_files:
        if not fname.endswith(".json"):
            continue  # skip non-JSON

        full_path = os.path.join(INPUT_ROOT_DIR, fname)

        # We check if it belongs to one of our known models and known prompt types
        # by substring match in the filename. (Alternatively parse the JSON inside.)
        # Let's see if it matches
        matched_model = None
        matched_prompt_type = None

        for model in MODEL_LS:
            # For file matching, we can replace ":" with "_" in the model
            # or simply see if the original "model" in some form is in the filename.
            # But the user has ":" in the model name, while the file name replaces ":" with something or underscores.
            # We'll do a conservative approach: remove or replace colons in both.
            # Actually in the example, the filenames do "athene-v2:72b-q4_K_M" as is.
            # We'll just check if model in fname:
            if model in fname:
                matched_model = model
                break

        # Now check prompt_type
        for pt in PROMPT_TYPE_LS:
            if pt in fname.lower():
                matched_prompt_type = pt
                break

        if matched_model and matched_prompt_type:
            # Good, we parse it
            logging.info(f"Processing file: {fname}")
            with open(full_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # We dispatch to the correct function
            # (Though the internal logic might be the same, we follow the instructions.)
            prompt_type_in_data = json_data.get("prompt_type", "").lower()
            if "system1" in prompt_type_in_data:
                row_dict = process_system1(json_data)
            elif "cot_nshot" in prompt_type_in_data or "cot-nshot" in prompt_type_in_data:
                row_dict = process_cot_nshot(json_data)
            elif "cot" in prompt_type_in_data:
                row_dict = process_cot(json_data)
            else:
                logging.warning(f"Unrecognized prompt_type in file: {fname}")
                continue

            csv_rows.append(row_dict)
        else:
            # Not matched
            logging.debug(f"Skipping file (no match): {fname}")

    # -----------------------------------------------------------------------------
    # 4. Write out the aggregated CSV
    # -----------------------------------------------------------------------------
    output_file = os.path.join(OUTPUT_ROOT_DIR, f'summary_all_{ENSEMBLE_NAME}.csv')
    logging.info(f"Writing aggregated CSV to: {output_file}")

    # We'll define the CSV columns in the order from the example.
    # Adjust as needed. This matches ###OUTPUT_CSV from the instructions:
    fieldnames = [
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
        "eval_count_missing_count",
    ]

    # First, sort the csv_rows list based on the model name
    # We assume the first fieldname corresponds to the model name column
    model_name_field = fieldnames[0]
    sorted_rows = sorted(csv_rows, key=lambda row: row[model_name_field].lower())

    with open(output_file, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)

    logging.info("Done.")


if __name__ == "__main__":
    main()
