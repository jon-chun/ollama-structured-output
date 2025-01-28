#!/usr/bin/env python3
"""
Enhanced model evaluation aggregation script with robust JSON parsing and regex fallback.
Handles both well-formed and malformed JSON files through multiple parsing strategies.
"""
import os
import json
import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

# --------------------------------------------------------------------
# 1. Global Constants and Configuration
# --------------------------------------------------------------------

EVALUATION_RESULTS_DIR = 'evaluation_results_long'
ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)
OUTPUT_DIR = os.path.join('aggregate_reports', os.path.basename(EVALUATION_RESULTS_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILENAME = f"aggregate_model_reports_claude_ver3.csv"

def setup_logging():
    """Configure detailed logging for better debugging of parsing issues."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'parsing_log_{timestamp}.txt'),
            logging.StreamHandler()
        ]
    )

def extract_value_with_type(value_str: str) -> Any:
    """
    Convert string values to appropriate Python types.
    Handles numbers, booleans, null/None, and strings.
    """
    # Remove any trailing commas and whitespace
    value_str = value_str.strip().rstrip(',').strip()
    
    # Handle null/None values
    if value_str.lower() in ('null', 'none'):
        return None

def aggregate_reports() -> pd.DataFrame:
    """
    Primary aggregation function that processes all report files and combines results.
    Includes enhanced error handling and data validation.
    """
    logging.info(f"Starting directory crawl from: {ROOT_DIR}")
    
    if not os.path.exists(ROOT_DIR):
        logging.error(f"Evaluation directory not found: {ROOT_DIR}")
        return pd.DataFrame()
    
    aggregated_rows = []
    processed_files = set()  # Track processed files to avoid duplicates
    
    for root, dirs, files in os.walk(ROOT_DIR):
        if os.path.basename(root) == 'reports':
            model_dir_name = os.path.basename(os.path.dirname(root))
            logging.info(f"Processing reports directory: {model_dir_name}")
            
            try:
                # Get model metadata
                model_meta = {
                    'model_dir': model_dir_name,
                    **extract_model_metadata(model_dir_name)
                }
                
                # Process JSON files
                json_files = [f for f in files if f.endswith('.json')]
                for jf in json_files:
                    if jf in processed_files:
                        continue
                        
                    # Find matching text file
                    tf = f"report_{jf.replace('metrics_', '').replace('.json', '.txt')}"
                    
                    if tf in files:
                        json_path = os.path.join(root, jf)
                        txt_path = os.path.join(root, tf)
                        
                        try:
                            combined_stats = process_report_files(json_path, txt_path)
                            if combined_stats:
                                combined_stats.update(model_meta)
                                aggregated_rows.append(combined_stats)
                                processed_files.add(jf)
                                logging.debug(f"Successfully processed {jf}")
                            else:
                                logging.warning(f"No data extracted from {jf}")
                        except Exception as e:
                            logging.error(f"Error processing files {jf}, {tf}: {str(e)}")
                            continue
                    else:
                        logging.warning(f"No matching TXT file found for {jf}")
                
            except Exception as e:
                logging.error(f"Error processing directory {root}: {str(e)}")
                continue
    
    # Create initial DataFrame
    df = pd.DataFrame(aggregated_rows)
    
    if df.empty:
        logging.warning("No data collected - returning empty DataFrame")
        return df
    
    # Fill missing metrics
    df = fill_missing_metrics(df)
    
    # Validate and clean data
    df = validate_dataframe(df)
    
    # Filter for required prompt types
    required_prompts = {'PromptType.COT', 'PromptType.COT_NSHOT', 'PromptType.SYSTEM1'}
    
    valid_models_data = []
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        model_prompts = set(model_df['prompt_type'].unique())
        
        if model_prompts >= required_prompts:
            model_rows = []
            for prompt_type in required_prompts:
                prompt_rows = model_df[model_df['prompt_type'] == prompt_type]
                if not prompt_rows.empty:
                    model_rows.append(prompt_rows.iloc[0])
            
            if len(model_rows) == len(required_prompts):
                valid_models_data.extend(model_rows)
                logging.info(f"Added complete prompt set for model: {model}")
            else:
                logging.warning(f"Incomplete prompt types for model {model}")
        else:
            logging.info(f"Skipping model {model} - missing prompt types")
    
    # Create final DataFrame
    final_df = pd.DataFrame(valid_models_data)
    
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
    
    for col in required_columns:
        if col not in final_df.columns:
            final_df[col] = None
    
    # Reorder columns to match required order
    final_df = final_df[required_columns]
    
    return final_df



def regex_parse_json(content: str) -> Dict[str, Any]:
    """
    Fallback JSON parser using regex to extract key-value pairs.
    Handles nested structures and malformed JSON.
    """
    logging.info("Using regex fallback parser")
    parsed_data = {}
    
    try:
        # Pattern for key-value pairs, handling various formats
        kv_pattern = r'"([^"]+)"\s*:\s*([^,}\]]+[,}\]]|"[^"]*"|{[^}]*}|\[[^\]]*\])'
        
        # Pattern for nested objects
        nested_pattern = r'{([^{}]*({[^{}]*})*[^{}]*)}'
        
        # Pattern for arrays
        array_pattern = r'\[(.*?)\]'
        
        def parse_nested_object(obj_str: str) -> Dict[str, Any]:
            """Parse a nested object string into a dictionary."""
            if not obj_str.strip():
                return {}
                
            nested_data = {}
            matches = re.finditer(kv_pattern, obj_str)
            
            for match in matches:
                key = match.group(1)
                value_str = match.group(2).strip()
                
                # Handle nested objects
                if value_str.startswith('{'):
                    nested_matches = re.search(nested_pattern, value_str)
                    if nested_matches:
                        value = parse_nested_object(nested_matches.group(1))
                    else:
                        value = {}
                # Handle arrays
                elif value_str.startswith('['):
                    array_matches = re.search(array_pattern, value_str)
                    if array_matches:
                        array_str = array_matches.group(1)
                        value = [extract_value_with_type(v.strip()) 
                                for v in array_str.split(',') if v.strip()]
                    else:
                        value = []
                else:
                    value = extract_value_with_type(value_str)
                
                nested_data[key] = value
            
            return nested_data
        
        # Extract meta_data section
        meta_pattern = r'"meta_data"\s*:\s*({[^}]*})'
        meta_match = re.search(meta_pattern, content)
        if meta_match:
            parsed_data['meta_data'] = parse_nested_object(meta_match.group(1))
        
        # Extract confusion_matrix section
        confusion_pattern = r'"confusion_matrix"\s*:\s*({[^}]*})'
        confusion_match = re.search(confusion_pattern, content)
        if confusion_match:
            parsed_data['confusion_matrix'] = parse_nested_object(confusion_match.group(1))
        
        # Extract other top-level key-value pairs
        for match in re.finditer(kv_pattern, content):
            key = match.group(1)
            if key not in ('meta_data', 'confusion_matrix'):
                value_str = match.group(2)
                parsed_data[key] = extract_value_with_type(value_str)
        
        return parsed_data
        
    except Exception as e:
        logging.error(f"Error in regex parsing: {str(e)}")
        return {}

def safe_json_load(file_path: str) -> Tuple[Dict[str, Any], bool]:
    """
    Attempt to load JSON file with multiple fallback strategies.
    Returns tuple of (parsed_data, is_malformed).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # First try: Standard JSON parsing
        try:
            data = json.loads(content)
            return data, False
        except json.JSONDecodeError as e:
            logging.warning(f"Standard JSON parsing failed: {str(e)}")
            
            # Second try: Clean common issues and retry JSON parsing
            try:
                # Remove comments
                content_clean = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.S)
                # Fix unquoted keys
                content_clean = re.sub(r'(\w+)(?=\s*:)', r'"\1"', content_clean)
                # Fix trailing commas
                content_clean = re.sub(r',(\s*[}\]])', r'\1', content_clean)
                
                data = json.loads(content_clean)
                return data, True
            except json.JSONDecodeError:
                logging.warning("Cleaned JSON parsing failed, falling back to regex parser")
                
                # Final try: Regex-based parsing
                data = regex_parse_json(content)
                return data, True
                
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return {}, True

def extract_metrics_from_malformed(content: str) -> Dict[str, Any]:
    """
    Extract specific metrics from malformed JSON using targeted regex patterns.
    """
    metrics = {}
    
    patterns = {
        'prediction_accuracy': r'"prediction_accuracy":\s*([\d.]+)',
        'auc_roc': r'"auc_roc":\s*([\d.]+)',
        'total_attempts': r'"total_attempts":\s*(\d+)',
        'successful_attempts': r'"successful_attempts":\s*(\d+)',
        'failed_attempts': r'"failed_attempts":\s*(\d+)',
        'timeout_attempts': r'"timeout_attempts":\s*(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            try:
                value = float(match.group(1))
                metrics[key] = value
            except ValueError:
                logging.warning(f"Could not convert {key} value to float")
                metrics[key] = None
    
    return metrics

def fill_missing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing metrics with appropriate default values or calculations.
    This ensures all required columns have valid data.
    """
    # Define default values for different metric types
    numeric_defaults = {
        'total_duration_mean': 0.0,
        'total_duration_median': 0.0,
        'total_duration_sd': 0.0,
        'load_duration_mean': 0.0,
        'load_duration_median': 0.0,
        'load_duration_sd': 0.0,
        'prompt_eval_duration_mean': 0.0,
        'prompt_eval_duration_median': 0.0,
        'prompt_eval_duration_sd': 0.0,
        'eval_duration_mean': 0.0,
        'eval_duration_median': 0.0,
        'eval_duration_sd': 0.0,
        'prompt_eval_count_mean': 0,
        'prompt_eval_count_median': 0,
        'prompt_eval_count_sd': 0,
        'eval_count_mean': 0,
        'eval_count_median': 0,
        'eval_count_sd': 0
    }
    
    # Fill missing numeric values
    for col, default in numeric_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
        else:
            df[col] = default
    
    # Ensure missing count columns exist and have valid values
    missing_count_columns = [
        'total_duration_sec_missing_count',
        'load_duration_sec_missing_count',
        'prompt_eval_duration_sec_missing_count',
        'eval_duration_sec_missing_count',
        'python_api_duration_sec_missing_count',
        'confidence_txt_missing_count',
        'prompt_eval_count_missing_count',
        'eval_count_missing_count'
    ]
    
    for col in missing_count_columns:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype(int)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the DataFrame to ensure data quality.
    Performs type conversion and validation of values.
    """
    # Define expected types for columns
    float_columns = [
        'execution_time_mean', 'execution_time_median', 'execution_time_sd',
        'prediction_accuracy', 'auc_roc', 'txt_accuracy', 'txt_auc_roc',
        'total_duration_mean', 'total_duration_median', 'total_duration_sd',
        'load_duration_mean', 'load_duration_median', 'load_duration_sd',
        'prompt_eval_duration_mean', 'prompt_eval_duration_median', 'prompt_eval_duration_sd',
        'eval_duration_mean', 'eval_duration_median', 'eval_duration_sd',
        'precision', 'recall', 'f1_score'
    ]
    
    int_columns = [
        'total_attempts', 'successful_attempts', 'failed_attempts', 'timeout_attempts',
        'true_positives', 'true_negatives', 'false_positives', 'false_negatives',
        'prompt_eval_count_mean', 'prompt_eval_count_median', 'prompt_eval_count_sd',
        'eval_count_mean', 'eval_count_median', 'eval_count_sd'
    ]
    
    # Convert and validate float columns
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace invalid values with 0.0
            df[col] = df[col].fillna(0.0)
            # Ensure values are within reasonable ranges
            df.loc[df[col] < 0, col] = 0.0
            df.loc[df[col] > 1e6, col] = 0.0  # Cap unreasonably large values
    
    # Convert and validate integer columns
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype(int)
            # Ensure non-negative values
            df.loc[df[col] < 0, col] = 0
    
    return df

def process_report_files(json_path: str, txt_path: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced report processing with robust JSON parsing and fallback mechanisms.
    Handles malformed JSON files and ensures consistent output format.
    """
    logging.info(f"Processing files => JSON: {json_path}, TXT: {txt_path}")
    
    try:
        # Attempt to load and parse JSON with fallback strategies
        jdata, is_malformed = safe_json_load(json_path)
        
        if not jdata:
            logging.error(f"Failed to parse JSON file: {json_path}")
            return None
            
        # If malformed, attempt to extract additional metrics
        if is_malformed:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
            additional_metrics = extract_metrics_from_malformed(content)
            jdata.update(additional_metrics)
        
        # Process meta_data with enhanced error handling
        meta_data = jdata.get('meta_data', {})
        if isinstance(meta_data, str):
            try:
                meta_data = json.loads(meta_data)
            except json.JSONDecodeError:
                meta_data = regex_parse_json(meta_data)
        
        # Extract all available metrics
        combined = {
            'model_name': jdata.get('model_name'),
            'prompt_type': jdata.get('prompt_type'),
            'total_attempts': jdata.get('total_attempts'),
            'successful_attempts': jdata.get('successful_attempts'),
            'failed_attempts': jdata.get('failed_attempts'),
            'timeout_attempts': jdata.get('timeout_attempts')
        }
        
        # Process additional metrics based on availability
        if isinstance(meta_data, dict):
            duration_metrics = ['total_duration', 'load_duration', 
                              'prompt_eval_duration', 'eval_duration']
            
            for metric in duration_metrics:
                value = meta_data.get(metric)
                if value is not None:
                    try:
                        value = float(value)
                        combined[f'{metric}_mean'] = value
                        combined[f'{metric}_median'] = value
                        combined[f'{metric}_sd'] = 0
                    except (ValueError, TypeError):
                        combined[f'{metric}_mean'] = None
                        combined[f'{metric}_median'] = None
                        combined[f'{metric}_sd'] = None
            
            # Process count metrics
            count_metrics = ['prompt_eval_count', 'eval_count']
            for metric in count_metrics:
                value = meta_data.get(metric)
                if value is not None:
                    try:
                        value = int(value)
                        combined[f'{metric}_mean'] = value
                        combined[f'{metric}_median'] = value
                        combined[f'{metric}_sd'] = 0
                    except (ValueError, TypeError):
                        combined[f'{metric}_mean'] = None
                        combined[f'{metric}_median'] = None
                        combined[f'{metric}_sd'] = None
        
        # Process confusion matrix
        confusion = jdata.get('confusion_matrix', {})
        if isinstance(confusion, str):
            try:
                confusion = json.loads(confusion)
            except json.JSONDecodeError:
                confusion = regex_parse_json(confusion)
        
        # Extract confusion matrix metrics
        tp = confusion.get('tp', 0)
        tn = confusion.get('tn', 0)
        fp = confusion.get('fp', 0)
        fn = confusion.get('fn', 0)
        
        combined.update({
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        })
        
        # Calculate derived metrics
        if tp + fp > 0:
            combined['precision'] = tp / (tp + fp)
        else:
            combined['precision'] = 0.0
            
        if tp + fn > 0:
            combined['recall'] = tp / (tp + fn)
        else:
            combined['recall'] = 0.0
            
        if combined['precision'] + combined['recall'] > 0:
            combined['f1_score'] = (2 * combined['precision'] * combined['recall'] /
                                  (combined['precision'] + combined['recall']))
        else:
            combined['f1_score'] = 0.0
        
        # Process TXT file
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_content = f.read()
                
            accuracy_match = re.search(r'Accuracy:\s*([\d.]+)%', txt_content)
            if accuracy_match:
                combined['txt_accuracy'] = float(accuracy_match.group(1))
                
            auc_match = re.search(r'AUC-ROC:\s*([\d.]+)', txt_content)
            if auc_match:
                combined['txt_auc_roc'] = float(auc_match.group(1))
        except Exception as e:
            logging.warning(f"Error processing TXT file {txt_path}: {str(e)}")
            combined['txt_accuracy'] = None
            combined['txt_auc_roc'] = None
        
        return combined
        
    except Exception as e:
        logging.error(f"Error processing files {json_path}, {txt_path}: {str(e)}")
        return None
    
def aggregate_reports() -> pd.DataFrame:
    """
    Main aggregation function that processes all report files and combines results.
    Includes comprehensive error handling and data validation.
    """
    logging.info(f"Starting directory crawl from: {ROOT_DIR}")
    
    if not os.path.exists(ROOT_DIR):
        logging.error(f"Evaluation directory not found: {ROOT_DIR}")
        return pd.DataFrame()
    
    aggregated_rows = []
    processed_files = set()  # Track processed files to avoid duplicates
    
    for root, dirs, files in os.walk(ROOT_DIR):
        if os.path.basename(root) == 'reports':
            model_dir_name = os.path.basename(os.path.dirname(root))
            logging.info(f"Processing reports directory: {model_dir_name}")
            
            try:
                # Extract model metadata
                model_meta = {
                    'model_dir': model_dir_name,
                    **extract_model_metadata(model_dir_name)
                }
                
                # Process all JSON files in directory
                json_files = [f for f in files if f.endswith('.json')]
                for jf in json_files:
                    if jf in processed_files:
                        continue
                    
                    # Find matching text file
                    tf = f"report_{jf.replace('metrics_', '').replace('.json', '.txt')}"
                    
                    if tf in files:
                        json_path = os.path.join(root, jf)
                        txt_path = os.path.join(root, tf)
                        
                        try:
                            # Process the file pair
                            combined_stats = process_report_files(json_path, txt_path)
                            if combined_stats:
                                combined_stats.update(model_meta)
                                aggregated_rows.append(combined_stats)
                                processed_files.add(jf)
                                logging.debug(f"Successfully processed {jf}")
                            else:
                                logging.warning(f"No valid data extracted from {jf}")
                        except Exception as e:
                            logging.error(f"Error processing files {jf}, {tf}: {str(e)}")
                            continue
                    else:
                        logging.warning(f"No matching TXT file found for {jf}")
            
            except Exception as e:
                logging.error(f"Error processing directory {root}: {str(e)}")
                continue
    
    # Create DataFrame from aggregated data
    df = pd.DataFrame(aggregated_rows)
    
    if df.empty:
        logging.warning("No data collected - returning empty DataFrame")
        return df
    
    # Filter for required prompt types and ensure data completeness
    required_prompts = {'PromptType.COT', 'PromptType.COT_NSHOT', 'PromptType.SYSTEM1'}
    
    valid_models_data = []
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        model_prompts = set(model_df['prompt_type'].unique())
        
        if model_prompts >= required_prompts:
            # Collect data for all required prompt types
            model_rows = []
            for prompt_type in required_prompts:
                prompt_rows = model_df[model_df['prompt_type'] == prompt_type]
                if not prompt_rows.empty:
                    model_rows.append(prompt_rows.iloc[0])
            
            # Only include models with complete data
            if len(model_rows) == len(required_prompts):
                valid_models_data.extend(model_rows)
                logging.info(f"Added complete prompt set for model: {model}")
            else:
                logging.warning(f"Incomplete prompt types for model {model}")
        else:
            logging.info(f"Skipping model {model} - missing required prompt types")
    
    # Create final DataFrame with validated data
    final_df = pd.DataFrame(valid_models_data)
    
    # Ensure all required columns are present with appropriate default values
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
    
    # Add missing columns with appropriate default values
    for col in required_columns:
        if col not in final_df.columns:
            if any(metric in col for metric in ['duration', 'time']):
                final_df[col] = 0.0
            elif 'count' in col:
                final_df[col] = 0
            else:
                final_df[col] = None
    
    # Reorder columns to match required order
    final_df = final_df[required_columns]
    
    return final_df

def main():
    """
    Main execution function that orchestrates the entire process of reading,
    aggregating, and saving model evaluation results.
    """
    logging.info("Starting model evaluation aggregation process...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Process all reports
        df = aggregate_reports()
        
        if df.empty:
            logging.error("No valid reports were found or processed")
            return
        
        # Sort by model name for consistency
        df.sort_values('model_name', inplace=True)
        
        # Save results to CSV
        output_path = os.path.join(OUTPUT_DIR, 'aggregate_model_reports_claude_ver2.csv')
        df.to_csv(output_path, index=False)
        logging.info(f"Results saved to: {output_path}")
        
        # Generate and log summary statistics
        total_models = len(df['model_name'].unique())
        total_rows = len(df)
        logging.info(f"Successfully processed {total_models} models, {total_rows} total rows")
        
        # Log data quality metrics
        null_counts = df.isnull().sum()
        if null_counts.any():
            logging.warning("Columns with missing values:")
            for col in null_counts[null_counts > 0].index:
                logging.warning(f"  {col}: {null_counts[col]} missing values")
        
        logging.info("Processing completed successfully!")
        
    except Exception as e:
        logging.critical(f"Fatal error in main execution: {str(e)}")
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


        