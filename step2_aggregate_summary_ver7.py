#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

# --------------------------------------------------------------------
# 1. Global Constants
# --------------------------------------------------------------------
EVALUATION_RESULTS_DIR = "evaluation_results_short_20250108"
INPUT_ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)

OUTPUT_DIR = os.path.join("aggregate_reports", os.path.basename(EVALUATION_RESULTS_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILENAME = "aggregate_model_reports.csv"

# Regex patterns for JSON field extraction
JSON_PATTERNS = {
    'ntop_text_summary': r'"ntop_text_summary":\s*"([^"]+)"',
    'age': r'"age":\s*(\d+)',
    'numberofarrestsby2002': r'"numberofarrestsby2002":\s*(\d+)',
    'y_arrestedafter2002': r'"y_arrestedafter2002":\s*(true|false)',
    'prediction': r'"prediction":\s*"([^"]+)"',
}

# --------------------------------------------------------------------
# 2. Setup Logging
# --------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"log_aggregate_reports_{timestamp}.txt"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s][%(levelname)s] %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

def logging_custom(level: str, message: str):
    """Custom logging function with standardized formatting."""
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
def extract_model_metadata(model_dir_name: str) -> Dict[str, str]:
    """Extract model parameters and quantization from directory name."""
    logging_custom("DEBUG", f"extract_model_metadata => directory: {model_dir_name}")

    param_match = re.search(r"(\d+)b", model_dir_name.lower())
    quant_match = re.search(r"(q4_k_m|fp16)", model_dir_name.lower())

    params = param_match.group(1) if param_match else "unknown"
    quant = quant_match.group(1) if quant_match else "unknown"

    return {
        "model_params": params,
        "model_quantization": quant,
    }

def extract_json_fields(content: str) -> Dict[str, Union[str, int, bool]]:
    """Extract specific fields from potentially malformed JSON using regex."""
    results = {}
    
    for field, pattern in JSON_PATTERNS.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            # Convert types appropriately
            if field in ['age', 'numberofarrestsby2002']:
                results[field] = int(value)
            elif field == 'y_arrestedafter2002':
                results[field] = value.lower() == 'true'
            else:
                results[field] = value
                
    return results

def safe_parse_json(content: str) -> Optional[dict]:
    """Attempt multiple JSON parsing strategies."""
    # 1. Standard parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # 2. Remove trailing commas
    try:
        fixed = re.sub(r",\s*([\]}])", r"\1", content)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # 3. Fallback to regex extraction
    return extract_json_fields(content)

def get_txt_for_json_filename(json_filename: str) -> str:
    """Map JSON filename to corresponding TXT filename."""
    base_name, _ = os.path.splitext(json_filename)
    if base_name.startswith("metrics_"):
        txt_name = "report_" + base_name[len("metrics_"):]
    else:
        txt_name = base_name
    return txt_name + ".txt"

# --------------------------------------------------------------------
# 4. File Parsing
# --------------------------------------------------------------------
def parse_metadata_api(content: str) -> Dict[str, float]:
    """
    Parse metadata API section with multiple fallback methods.
    Returns duration values in seconds.
    """
    metadata = {}
    
    # Try direct JSON parsing first
    try:
        api_section = re.search(r"Metadata API:\s*({[^}]+})", content)
        if api_section:
            json_str = api_section.group(1)
            data = json.loads(json_str)
            
            # Extract duration fields
            for key in ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']:
                sec_key = f"{key}_sec"
                ns_key = f"{key}_ns"
                
                if sec_key in data:
                    metadata[key] = float(data[sec_key])
                elif ns_key in data:
                    metadata[key] = float(data[ns_key]) / 1e9
                
            return metadata
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fallback to regex extraction
    patterns = {
        'total_duration': r'total_duration_(?:sec|ns).*?([0-9.]+)',
        'load_duration': r'load_duration_(?:sec|ns).*?([0-9.]+)',
        'prompt_eval_duration': r'prompt_eval_duration_(?:sec|ns).*?([0-9.]+)',
        'eval_duration': r'eval_duration_(?:sec|ns).*?([0-9.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val = float(match.group(1))
            # Convert if in nanoseconds
            if '_ns' in match.group(0):
                val /= 1e9
            metadata[key] = val
            
    return metadata

def parse_report_txt(txt_path: str) -> Dict[str, Union[float, str]]:
    """Parse TXT report file with enhanced error handling."""
    summary = {}
    
    if not os.path.exists(txt_path):
        logging_custom("WARNING", f"TXT file missing: {txt_path}")
        return summary
        
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse basic metrics
        if 'Judge Prediction:' in content:
            pred = re.search(r'Judge Prediction:\s*(\w+)', content)
            summary['judge_prediction'] = pred.group(1) if pred else "UNKNOWN"
            
        if 'Accuracy:' in content:
            acc = re.search(r'Accuracy:\s*([\d.]+)%?', content)
            summary['txt_accuracy'] = float(acc.group(1)) if acc else None
            
        if 'AUC-ROC:' in content:
            auc = re.search(r'AUC-ROC:\s*([\d.]+)', content)
            summary['txt_auc_roc'] = float(auc.group(1)) if auc else None
            
        # Parse metadata API section
        metadata = parse_metadata_api(content)
        summary.update(metadata)
        
    except Exception as e:
        logging_custom("ERROR", f"Error parsing TXT {txt_path}: {e}")
        
    return summary

def process_report_files(json_path: str, txt_path: str, model_dir_name: str) -> List[Dict]:
    """Process a pair of report files with enhanced error handling."""
    logging_custom("INFO", f"Processing files => JSON: {json_path}, TXT: {txt_path}")
    
    # Read and parse JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = f.read()
            
        jdata = safe_parse_json(json_content)
        if not jdata:
            logging_custom("ERROR", f"Failed to parse JSON: {json_path}")
            return []
    except Exception as e:
        logging_custom("ERROR", f"Error reading JSON {json_path}: {e}")
        return []
        
    # Parse TXT companion file
    tdata = parse_report_txt(txt_path)
    
    # Extract model metadata
    model_meta = extract_model_metadata(model_dir_name)
    
    # Handle special "No decision" case
    prediction = jdata.get('prediction', '')
    if prediction == 'No decision':
        tdata['judge_prediction'] = "UNKNOWN"
        jdata['prediction_accurate'] = "UNKNOWN"
    
    # Build output rows
    out_rows = []
    
    # Add basic metrics
    base_row = {
        'model_name': jdata.get('model_name', 'unknown'),
        'prompt_type': jdata.get('prompt_type', 'unknown'),
        'total_attempts': jdata.get('total_attempts'),
        'successful_attempts': jdata.get('successful_attempts'),
        'failed_attempts': jdata.get('failed_attempts'),
        'timeout_attempts': jdata.get('timeout_attempts'),
        'prediction_accuracy': jdata.get('prediction_accuracy'),
        'auc_roc': jdata.get('auc_roc'),
        
        # Add confusion matrix data
        'true_positives': jdata.get('confusion_matrix', {}).get('tp', 0),
        'true_negatives': jdata.get('confusion_matrix', {}).get('tn', 0),
        'false_positives': jdata.get('confusion_matrix', {}).get('fp', 0),
        'false_negatives': jdata.get('confusion_matrix', {}).get('fn', 0),
        
        # Add TXT file metrics
        'txt_accuracy': tdata.get('txt_accuracy'),
        'txt_auc_roc': tdata.get('txt_auc_roc'),
        
        # Add model metadata
        'model_dir': model_dir_name,
        'model_params': model_meta['model_params'],
        'model_quantization': model_meta['model_quantization'],
    }
    
    # Add metadata timings if available
    if any(k in tdata for k in ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']):
        base_row.update({
            'total_duration': tdata.get('total_duration'),
            'load_duration': tdata.get('load_duration'),
            'prompt_eval_duration': tdata.get('prompt_eval_duration'),
            'eval_duration': tdata.get('eval_duration'),
        })
    
    out_rows.append(base_row)
    return out_rows

# --------------------------------------------------------------------
# 5. Core Aggregation
# --------------------------------------------------------------------
def aggregate_reports() -> pd.DataFrame:
    """Main aggregation function with enhanced error handling."""
    logging_custom("INFO", f"Starting aggregation from: {INPUT_ROOT_DIR}")
    
    if not os.path.exists(INPUT_ROOT_DIR):
        logging_custom("ERROR", f"Input directory not found: {INPUT_ROOT_DIR}")
        return pd.DataFrame()
        
    all_raw_rows = []
    
    # Walk directory tree
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        logging_custom("DEBUG", f"Scanning directory: {root}")
        
        # Process model directories directly
        model_dir_name = os.path.basename(root)
        
        # Find JSON/TXT pairs
        json_files = [f for f in files if f.endswith('.json')]
        for jf in json_files:
            tf = get_txt_for_json_filename(jf)
            json_path = os.path.join(root, jf)
            txt_path = os.path.join(root, tf)
            
            if os.path.exists(txt_path):
                rows = process_report_files(json_path, txt_path, model_dir_name)
                all_raw_rows.extend(rows)
            else:
                logging_custom("WARNING", f"Missing TXT file for {jf}")
                
    # Convert to DataFrame
    df = pd.DataFrame(all_raw_rows)
    if df.empty:
        logging_custom("WARNING", "No data found")
        return df
        
    # Compute derived metrics
    df['precision'] = df.apply(
        lambda r: r['true_positives'] / (r['true_positives'] + r['false_positives']) 
        if r['true_positives'] + r['false_positives'] > 0 else 0.0,
        axis=1
    )
    
    df['recall'] = df.apply(
        lambda r: r['true_positives'] / (r['true_positives'] + r['false_negatives'])
        if r['true_positives'] + r['false_negatives'] > 0 else 0.0,
        axis=1
    )
    
    df['f1_score'] = df.apply(
        lambda r: 2 * (r['precision'] * r['recall']) / (r['precision'] + r['recall'])
        if r['precision'] + r['recall'] > 0 else 0.0,
        axis=1
    )
    
    return df

# --------------------------------------------------------------------
# 6. Main Execution
# --------------------------------------------------------------------
def main():
    """Main execution with enhanced error handling and data validation."""
    logging_custom("INFO", "Starting report aggregation...")
    
    try:
        # Run main aggregation
        df = aggregate_reports()
        
        if df.empty:
            logging_custom("WARNING", "No valid data found to aggregate")
            return
            
        # Ensure all required columns exist
        required_cols = [
            "model_name", "prompt_type", 
            "total_attempts", "successful_attempts", "failed_attempts",
            "prediction_accuracy", "auc_roc",
            "true_positives", "true_negatives", "false_positives", "false_negatives",
            "precision", "recall", "f1_score",
            "txt_accuracy", "txt_auc_roc",
            "model_params", "model_quantization"
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
                logging_custom("WARNING", f"Missing column '{col}' added with NULL values")
                
        # Convert numeric columns
        numeric_cols = [
            "total_attempts", "successful_attempts", "failed_attempts",
            "prediction_accuracy", "auc_roc", "precision", "recall", "f1_score",
            "txt_accuracy", "txt_auc_roc",
            "total_duration", "load_duration", "prompt_eval_duration", "eval_duration"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                nulls = df[col].isna().sum()
                if nulls > 0:
                    logging_custom("WARNING", f"Column '{col}' has {nulls} NULL values")
                    
        # Calculate aggregated statistics
        agg_df = df.groupby(['model_name', 'prompt_type'], as_index=False).agg({
            'total_attempts': 'sum',
            'successful_attempts': 'sum',
            'failed_attempts': 'sum',
            'prediction_accuracy': 'mean',
            'auc_roc': 'mean',
            'total_duration': ['mean', 'median', 'std'],
            'load_duration': ['mean', 'median', 'std'],
            'prompt_eval_duration': ['mean', 'median', 'std'],
            'eval_duration': ['mean', 'median', 'std'],
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'txt_accuracy': 'mean',
            'txt_auc_roc': 'mean',
            'model_params': 'first',
            'model_quantization': 'first'
        })
        
        # Flatten column names
        agg_df.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
            for col in agg_df.columns
        ]
        
        # Replace _std with _sd for consistency
        agg_df.columns = [
            col.replace('_std', '_sd') for col in agg_df.columns
        ]
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        agg_df.to_csv(output_path, index=False)
        logging_custom("INFO", f"Successfully saved aggregated results to {output_path}")
        logging_custom("INFO", f"Processed {len(df)} raw rows into {len(agg_df)} aggregated rows")
        
    except Exception as e:
        logging_custom("ERROR", f"Aggregation failed: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()