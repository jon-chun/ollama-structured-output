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
from typing import Dict, List, Optional, Any, Tuple

# --------------------------------------------------------------------
# 1. Global Constants and Configuration
# --------------------------------------------------------------------

EVALUATION_RESULTS_DIR = 'evaluation_results_long'
ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)
OUTPUT_DIR = os.path.join('aggregate_reports', os.path.basename(EVALUATION_RESULTS_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILENAME = f"aggregate_model_reports_claude_ver1.csv"

# --------------------------------------------------------------------
# 2. Enhanced Logging Setup
# --------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    Returns configured logger.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f"log_aggregate_reports_{timestamp}.txt"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler (DEBUG-level messages)
    fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler (INFO-level messages)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Enhanced formatter with thread info
    formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Initialize logger
logger = setup_logging()

def logging_custom(level: str, message: str):
    """Enhanced custom logging with better error handling."""
    try:
        level = level.upper().strip()
        log_func = getattr(logger, level.lower(), None)
        if log_func:
            log_func(message)
        else:
            logger.warning(f"Invalid log level '{level}', defaulting to DEBUG. Message: {message}")
            logger.debug(message)
    except Exception as e:
        logger.error(f"Logging error: {str(e)}. Original message: {message}")

# --------------------------------------------------------------------
# 3. Enhanced Utility Functions
# --------------------------------------------------------------------
def extract_model_metadata(model_dir_name: str) -> Dict[str, str]:
    """
    Extract model size and quantization info from directory name with enhanced validation.
    """
    logging_custom("DEBUG", f"extract_model_metadata => directory: {model_dir_name}")
    
    metadata = {
        'model_params': "unknown",
        'model_quantization': "unknown"
    }
    
    try:
        param_match = re.search(r'(\d+)b', model_dir_name.lower())
        quant_match = re.search(r'(q4_k_m|fp16|q4_k_m)', model_dir_name.lower())
        
        if param_match:
            metadata['model_params'] = param_match.group(1)
        if quant_match:
            metadata['model_quantization'] = quant_match.group(1)
            
        logging_custom("DEBUG", 
                      f"Extracted metadata: params={metadata['model_params']}, "
                      f"quant={metadata['model_quantization']}")
    except Exception as e:
        logging_custom("ERROR", f"Error extracting model metadata: {str(e)}")
    
    return metadata

def clean_model_name(model_name: str) -> str:
    """
    Convert model name to OS-friendly format with validation.
    """
    if not isinstance(model_name, str):
        logging_custom("WARNING", f"Invalid model name type: {type(model_name)}")
        return ""
    
    try:
        cleaned = model_name.strip().lower()
        cleaned = re.sub(r'[:.]+', '_', cleaned)
        cleaned = re.sub(r'[^\w]+', '_', cleaned)
        cleaned = re.sub(r'_+', '_', cleaned)
        cleaned = cleaned.strip('_')
        return cleaned
    except Exception as e:
        logging_custom("ERROR", f"Error cleaning model name: {str(e)}")
        return model_name

def calculate_missing_counts(meta_data: List[Dict]) -> Dict[str, int]:
    """
    Calculate missing value counts for various duration metrics.
    """
    counts = {
        'total_duration_sec_missing_count': 0,
        'load_duration_sec_missing_count': 0,
        'prompt_eval_duration_sec_missing_count': 0,
        'eval_duration_sec_missing_count': 0,
        'python_api_duration_sec_missing_count': 0,
        'confidence_txt_missing_count': 0,
        'prompt_eval_count_missing_count': 0,
        'eval_count_missing_count': 0
    }
    
    for entry in meta_data:
        if not entry.get('total_duration'):
            counts['total_duration_sec_missing_count'] += 1
        if not entry.get('load_duration'):
            counts['load_duration_sec_missing_count'] += 1
        if not entry.get('prompt_eval_duration'):
            counts['prompt_eval_duration_sec_missing_count'] += 1
        if not entry.get('eval_duration'):
            counts['eval_duration_sec_missing_count'] += 1
        if not entry.get('python_api_duration'):
            counts['python_api_duration_sec_missing_count'] += 1
        if not entry.get('confidence'):
            counts['confidence_txt_missing_count'] += 1
        if not entry.get('prompt_eval_count'):
            counts['prompt_eval_count_missing_count'] += 1
        if not entry.get('eval_count'):
            counts['eval_count_missing_count'] += 1
    
    return counts

# --------------------------------------------------------------------
# 4. Enhanced Data Processing Functions
# --------------------------------------------------------------------
def calculate_count_statistics(meta_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate statistics for prompt_eval_count and eval_count.
    """
    prompt_eval_counts = []
    eval_counts = []
    
    for entry in meta_data:
        if isinstance(entry, dict):
            if 'prompt_eval_count' in entry:
                prompt_eval_counts.append(entry['prompt_eval_count'])
            if 'eval_count' in entry:
                eval_counts.append(entry['eval_count'])
    
    stats = {}
    
    # Calculate prompt_eval_count statistics
    if prompt_eval_counts:
        stats.update({
            'prompt_eval_count_mean': np.mean(prompt_eval_counts),
            'prompt_eval_count_median': np.median(prompt_eval_counts),
            'prompt_eval_count_sd': np.std(prompt_eval_counts) if len(prompt_eval_counts) > 1 else 0
        })
    else:
        stats.update({
            'prompt_eval_count_mean': None,
            'prompt_eval_count_median': None,
            'prompt_eval_count_sd': None
        })
    
    # Calculate eval_count statistics
    if eval_counts:
        stats.update({
            'eval_count_mean': np.mean(eval_counts),
            'eval_count_median': np.median(eval_counts),
            'eval_count_sd': np.std(eval_counts) if len(eval_counts) > 1 else 0
        })
    else:
        stats.update({
            'eval_count_mean': None,
            'eval_count_median': None,
            'eval_count_sd': None
        })
    
    return stats

def process_report_files(json_path: str, txt_path: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced processing of report files with additional metrics and error handling.
    """
    logging_custom("DEBUG", f"Processing files => JSON: {json_path}, TXT: {txt_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            jdata = json.load(f)
    except Exception as e:
        logging_custom("ERROR", f"Error reading JSON {json_path}: {str(e)}")
        return None
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            tdata = {
                'txt_accuracy': None,
                'txt_auc_roc': None
            }
            for line in f:
                if "Accuracy:" in line:
                    try:
                        tdata['txt_accuracy'] = float(line.split(":")[-1].strip().replace('%', ''))
                    except ValueError:
                        pass
                elif "AUC-ROC:" in line:
                    try:
                        tdata['txt_auc_roc'] = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
    except Exception as e:
        logging_custom("ERROR", f"Error reading TXT {txt_path}: {str(e)}")
        tdata = {}
    
    # Initialize the combined dictionary with all required fields
    combined = {
        'model_name': jdata.get('model_name'),
        'prompt_type': jdata.get('prompt_type'),
        'total_attempts': jdata.get('total_attempts'),
        'successful_attempts': jdata.get('successful_attempts'),
        'failed_attempts': jdata.get('failed_attempts'),
        'timeout_attempts': jdata.get('timeout_attempts'),
        'execution_time_mean': jdata.get('avg_execution_time'),
        'execution_time_median': jdata.get('median_execution_time'),
        'execution_time_sd': jdata.get('sd_execution_time'),
        'prediction_accuracy': jdata.get('prediction_accuracy'),
        'auc_roc': jdata.get('auc_roc')
    }
    
    # Process meta_data for various statistics
    meta_data = jdata.get('meta_data', [])
    if isinstance(meta_data, dict):
        meta_data = [meta_data]  # Convert single dict to list
    
    # Calculate duration statistics
    duration_types = ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']
    for dtype in duration_types:
        values = [entry.get(dtype) for entry in meta_data if isinstance(entry, dict)]
        values = [v for v in values if v is not None]
        
        if values:
            combined[f'{dtype}_mean'] = np.mean(values)
            combined[f'{dtype}_median'] = np.median(values)
            combined[f'{dtype}_sd'] = np.std(values) if len(values) > 1 else 0
        else:
            combined[f'{dtype}_mean'] = None
            combined[f'{dtype}_median'] = None
            combined[f'{dtype}_sd'] = None
    
    # Add count statistics
    combined.update(calculate_count_statistics(meta_data))
    
    # Add missing value counts
    combined.update(calculate_missing_counts(meta_data))
    
    # Process confusion matrix
    confusion = jdata.get('confusion_matrix', {})
    combined.update({
        'true_positives': confusion.get('tp', 0),
        'true_negatives': confusion.get('tn', 0),
        'false_positives': confusion.get('fp', 0),
        'false_negatives': confusion.get('fn', 0)
    })
    
    # Calculate derived metrics
    tp = combined['true_positives']
    fp = combined['false_positives']
    fn = combined['false_negatives']
    
    combined['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    combined['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    combined['f1_score'] = 2 * (combined['precision'] * combined['recall']) / (combined['precision'] + combined['recall']) if (combined['precision'] + combined['recall']) > 0 else 0.0
    
    # Add text-based metrics
    combined.update(tdata)
    
    return combined

# --------------------------------------------------------------------
# 5. Enhanced Aggregation Function
# --------------------------------------------------------------------
def aggregate_reports() -> pd.DataFrame:
    """
    Enhanced report aggregation with better error handling and validation.
    """
    logging_custom("INFO", f"Starting directory crawl from: {ROOT_DIR}")
    
    if not os.path.exists(ROOT_DIR):
        logging_custom("ERROR", f"Evaluation directory not found: {ROOT_DIR}")
        return pd.DataFrame()
    
    aggregated_rows = []
    
    for root, dirs, files in os.walk(ROOT_DIR):
        if os.path.basename(root) == 'reports':
            model_dir_name = os.path.basename(os.path.dirname(root))
            logging_custom("INFO", f"Processing reports for model: {model_dir_name}")
            
            try:
                model_meta = {
                    'model_dir': model_dir_name,
                    **extract_model_metadata(model_dir_name)
                }
                
                json_files = [f for f in files if f.endswith('.json')]
                
                for jf in json_files:
                    tf = f"report_{jf.replace('metrics_', '').replace('.json', '.txt')}"
                    
                    if tf in files:
                        json_path = os.path.join(root, jf)
                        txt_path = os.path.join(root, tf)
                        
                        combined_stats = process_report_files(json_path, txt_path)
                        if combined_stats:
                            combined_stats.update(model_meta)
                            aggregated_rows.append(combined_stats)
                            logging_custom("DEBUG", f"Processed {jf} successfully")
                    else:
                        logging_custom("WARNING", f"Missing TXT file for {jf}")
                
            except Exception as e:
                logging_custom("ERROR", f"Error processing directory {root}: {str(e)}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(aggregated_rows)
    
    if df.empty:
        logging_custom("WARNING", "No data collected - empty DataFrame")
        return df
    
    # Filter for required prompt types
    required_prompts = {'PromptType.COT', 'PromptType.COT_NSHOT', 'PromptType.SYSTEM1'}
    
    valid_models_data = []
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        model_prompts = set(model_df['prompt_type'].unique())
        
        # Check if model has all required prompt types
        if model_prompts >= required_prompts:
            model_rows = []
            
            # For each prompt type, take the most recent occurrence
            for prompt_type in required_prompts:
                prompt_rows = model_df[model_df['prompt_type'] == prompt_type]
                if not prompt_rows.empty:
                    # Add the first occurrence for this prompt type
                    model_rows.append(prompt_rows.iloc[0])
            
            # Verify we have all required prompts
            if len(model_rows) == len(required_prompts):
                valid_models_data.extend(model_rows)
                logging_custom("INFO", f"Added complete prompt set for model: {model}")
            else:
                logging_custom("WARNING", 
                    f"Incomplete prompt types for model {model}. "
                    f"Found {len(model_rows)} of {len(required_prompts)} required types")
        else:
            logging_custom("INFO", 
                f"Skipping model {model} - missing prompt types: "
                f"{required_prompts - model_prompts}")
    
    # Create final DataFrame with only valid data
    final_df = pd.DataFrame(valid_models_data)
    
    if not final_df.empty:
        # Ensure all required columns are present
        required_columns = [
            'model_name', 'prompt_type', 'total_attempts', 'successful_attempts',
            'failed_attempts', 'timeout_attempts', 'execution_time_mean',
            'execution_time_median', 'execution_time_sd', 'prediction_accuracy',
            'auc_roc', 'txt_accuracy', 'txt_auc_roc', 'total_duration_mean',
            'total_duration_median', 'total_duration_sd', 'load_duration_mean',
            'load_duration_median', 'load_duration_sd', 'prompt_eval_duration_mean',
            'prompt_eval_duration_median', 'prompt_eval_duration_sd',
            'eval_duration_mean', 'eval_duration_median', 'eval_duration_sd',
            'prompt_eval_count_mean', 'prompt_eval_count_median',
            'prompt_eval_count_sd', 'eval_count_mean', 'eval_count_median',
            'eval_count_sd', 'true_positives', 'true_negatives', 'false_positives',
            'false_negatives', 'precision', 'recall', 'f1_score', 'model_dir',
            'model_params', 'model_quantization', 'total_duration_sec_missing_count',
            'load_duration_sec_missing_count', 'prompt_eval_duration_sec_missing_count',
            'eval_duration_sec_missing_count', 'python_api_duration_sec_missing_count',
            'confidence_txt_missing_count', 'prompt_eval_count_missing_count',
            'eval_count_missing_count'
        ]
        
        # Add any missing columns with None values
        for col in required_columns:
            if col not in final_df.columns:
                logging_custom("WARNING", f"Adding missing column: {col}")
                final_df[col] = None
                
        # Reorder columns to match required order
        final_df = final_df[required_columns]
        
        logging_custom("INFO", 
            f"Final dataset: {len(final_df)} rows, "
            f"{len(final_df['model_name'].unique())} unique models")
    else:
        logging_custom("WARNING", "No valid data after filtering")
    
    return final_df

# --------------------------------------------------------------------
# 6. Main Execution with Enhanced Error Handling
# --------------------------------------------------------------------
def save_dataframe(df: pd.DataFrame, output_path: str, format: str = 'csv'):
    """
    Save DataFrame with proper error handling and logging.
    """
    try:
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logging_custom("INFO", f"Successfully saved data to: {output_path}")
        logging_custom("DEBUG", f"Saved {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        logging_custom("ERROR", f"Failed to save data to {output_path}: {str(e)}")
        raise

def generate_summary_statistics(df: pd.DataFrame):
    """
    Generate and log summary statistics for key metrics.
    """
    try:
        stats = {
            'total_models': len(df['model_name'].unique()),
            'total_rows': len(df),
            'avg_accuracy': df['prediction_accuracy'].mean(),
            'avg_auc_roc': df['auc_roc'].mean(),
            'avg_f1_score': df['f1_score'].mean()
        }
        
        logging_custom("INFO", "Summary Statistics:")
        for key, value in stats.items():
            logging_custom("INFO", f"  {key}: {value:.2f}")
            
        return stats
    except Exception as e:
        logging_custom("ERROR", f"Error generating summary statistics: {str(e)}")
        return None

def main():
    """
    Main execution function with enhanced error handling and logging.
    """
    logging_custom("INFO", "Starting model evaluation aggregation process...")
    logging_custom("INFO", f"Reading from directory: {ROOT_DIR}")
    logging_custom("INFO", f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Aggregate reports
        df = aggregate_reports()
        
        if df.empty:
            logging_custom("ERROR", "No valid reports were found or processed")
            return
        
        # Sort by model name for consistency
        df.sort_values('model_name', inplace=True)
        
        # Generate summary statistics
        stats = generate_summary_statistics(df)
        
        # Save results
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        save_dataframe(df, output_path, format='json')
        
        # Optional: Save as CSV as well
        csv_path = output_path.replace('.json', '.csv')
        save_dataframe(df, csv_path, format='csv')
        
        logging_custom("INFO", "Processing completed successfully!")
        
    except Exception as e:
        logging_custom("CRITICAL", f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging_custom("INFO", "Process interrupted by user")
    except Exception as e:
        logging_custom("CRITICAL", f"Unhandled exception: {str(e)}")
        raise
        model_df = df[df['model_name'] == model]