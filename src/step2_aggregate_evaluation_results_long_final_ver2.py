import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

def ns2sec(nanoseconds: float) -> float:
    """Convert nanoseconds to seconds"""
    return nanoseconds / 1_000_000_000

def calculate_f1_score(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate F1 score from confusion matrix values.
    
    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
        
    Returns:
        F1 score as a float between 0 and 1, or 0 if undefined
    """
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
        
    return f1_score

def extract_metrics_from_json(json_path: Path) -> Dict:
    """
    Extract relevant metrics from a JSON report file and format them for CSV output.
    
    Args:
        json_path: Path to the JSON metrics file
        
    Returns:
        Dictionary containing formatted metrics for CSV output
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract model name and prompt type from filename
    # Example filename: metrics_aya-expanse:8b-q4_K_M_cot_20250126_144313.json
    filename_parts = json_path.stem.split('_')  # stem removes .json extension
    model_name = filename_parts[1]  # Get the model name part
    prompt_type = [part for part in filename_parts if part in ['system1', 'cot', 'cot-nshot']][0]
    
    # Calculate F1 score from confusion matrix values
    f1_score = calculate_f1_score(
        data['confusion_matrix']['tp'],
        data['confusion_matrix']['tn'],
        data['confusion_matrix']['fp'],
        data['confusion_matrix']['fn']
    )
    
    return {
        'model_name': model_name,
        'prompt_type': prompt_type.lower(),
        'accuracy': data['prediction_accuracy'],
        'f1_score': f1_score,
        'prediction_yes_ct': data['prediction_distribution']['YES'],
        'prediction_no_ct': data['prediction_distribution']['NO'],
        'actual_yes_ct': data['actual_distribution']['YES'],
        'actual_no_ct': data['actual_distribution']['NO'],
        'confusion_tp': data['confusion_matrix']['tp'],
        'confusion_tn': data['confusion_matrix']['tn'],
        'confusion_fp': data['confusion_matrix']['fp'],
        'confusion_fn': data['confusion_matrix']['fn'],
        'auc_roc': data['auc_roc'],
        'api_call_total_ct': data['total_attempts'],
        'api_call_success_ct': data['successful_attempts'],
        'api_total_duration_sec': ns2sec(data['meta_data_averages']['total_duration']),
        'api_prompt_eval_count': data['meta_data_averages']['prompt_eval_count'],
        'api_eval_count': data['meta_data_averages']['eval_count']
    }

def find_report_jsons(root_dir: str) -> List[Path]:
    """
    Find all metrics JSON files in report directories under the root directory.
    
    Args:
        root_dir: Root directory to search for report subdirectories
        
    Returns:
        List of paths to metrics JSON files
    """
    json_files = []
    root_path = Path(root_dir)
    
    # Find all directories ending with '_reports'
    report_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.endswith('_reports')]
    
    # Find all JSON files in report directories
    for report_dir in report_dirs:
        json_files.extend([f for f in report_dir.glob('*.json') if f.name.startswith('metrics_')])
    
    return json_files

def main(): 
    """
    Main function to aggregate evaluation results and create a summary CSV file.
    """
    # Define input and output paths
    # INPUT_ROOTDIR = 'evaluation_results_long_final_FREEZE'
    INPUT_ROOTDIR_NAME = 'evaluation_results_long_final_seed7_20250128'
    INPUT_ROOTDIR_PATH = os.path.join('..', INPUT_ROOTDIR_NAME)
    OUTPUT_DIR = os.path.join('..','aggregation_summary')
    OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, f'aggregation_{INPUT_ROOTDIR_NAME}.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all metrics JSON files
    json_files = find_report_jsons(INPUT_ROOTDIR_PATH)
    
    # Extract metrics from each JSON file
    all_metrics = []
    for json_file in json_files:
        try:
            metrics = extract_metrics_from_json(json_file)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Convert to DataFrame and save to CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # Sort by model_name and prompt_type for better organization
        df = df.sort_values(['model_name', 'prompt_type'])
        df.to_csv(OUTPUT_FILEPATH, index=False)
        print(f"Successfully created summary report: {OUTPUT_FILEPATH}")
        print(f"Processed {len(all_metrics)} evaluation files")
    else:
        print("No metrics files were found or successfully processed")

if __name__ == "__main__":
    main()