import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Global constants
EVALUATION_RESULTS_DIR = 'evaluation_results_long_20250110'
ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)  # Get absolute path
OUTPUT_DIR = os.path.join('data', 'aggregate_reports', os.path.basename(EVALUATION_RESULTS_DIR))

# Ensure the evaluation directory exists
if not os.path.exists(ROOT_DIR):
    raise FileNotFoundError(f"Evaluation directory not found: {ROOT_DIR}")

def extract_model_metadata(model_dir):
    """Extract model size and quantization information from directory name."""
    # Parse model parameters (e.g., 7b, 32b) and quantization (e.g., q4_k_m, fp16)
    param_match = re.search(r'(\d+)b', model_dir.lower())
    quant_match = re.search(r'(q4_k_m|fp16)', model_dir.lower())
    
    params = param_match.group(1) if param_match else "unknown"
    quant = quant_match.group(1) if quant_match else "unknown"
    
    return {
        'model_params_b': params,
        'quantization': quant
    }

def process_report_files(json_path, txt_path):
    """Process a pair of report files and return combined statistics."""
    try:
        # Read JSON report
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        print(f"Successfully processed: {json_path}")
    except:
        print(f"Failed to process: {json_path}")
        return None
    
    # Extract relevant metrics
    stats = {
        'model_name': json_data['model_name'],
        'prompt_type': json_data['prompt_type'],
        'total_attempts': json_data['total_attempts'],
        'successful_attempts': json_data['successful_attempts'],
        'failed_attempts': json_data['failed_attempts'],
        'avg_execution_time': json_data['avg_execution_time'],
        'prediction_accuracy': json_data['prediction_accuracy'],
        'auc_roc': json_data['auc_roc'],
        'avg_total_duration': json_data['meta_data_averages']['total_duration'],
        'avg_load_duration': json_data['meta_data_averages']['load_duration'],
        'avg_prompt_eval_duration': json_data['meta_data_averages']['prompt_eval_duration'],
        'avg_eval_duration': json_data['meta_data_averages']['eval_duration'],
    }
    
    # Add confusion matrix metrics
    stats.update({
        'true_positives': json_data['confusion_matrix']['tp'],
        'true_negatives': json_data['confusion_matrix']['tn'],
        'false_positives': json_data['confusion_matrix']['fp'],
        'false_negatives': json_data['confusion_matrix']['fn']
    })
    
    # Calculate additional metrics
    stats['precision'] = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
    stats['recall'] = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
    stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
    
    return stats

def aggregate_reports():
    """Crawl directory tree and aggregate all report files into a DataFrame."""
    print(f"Starting directory crawl from: {ROOT_DIR}")
    report_data = []
    reports_found = False  # Flag to track if we find any report directories
    
    # Walk through directory tree
    for root, dirs, files in os.walk(ROOT_DIR):
        if os.path.basename(root) == 'reports':
            # Get parent directory for model metadata
            model_dir = os.path.basename(os.path.dirname(root))
            model_meta = extract_model_metadata(model_dir)
            
            # Process all report file pairs in directory
            reports_found = True
            print(f"Found reports directory: {root}")
            print(f"Files in directory: {files}")
            
            json_files = [f for f in files if f.endswith('.json')]
            print(f"JSON files found: {json_files}")
            
            for json_file in json_files:
                base_name = json_file.replace('.json', '')
                txt_file = base_name + '.txt'
                
                if txt_file in files:
                    json_path = os.path.join(root, json_file)
                    txt_path = os.path.join(root, txt_file)
                    
                    # Process report pair and add model metadata
                    stats = process_report_files(json_path, txt_path)
                    if stats is not None:  # Only add successful processing results
                        stats.update(model_meta)
                        report_data.append(stats)
                        print(f"Added report data for: {json_file}")
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save DataFrame
    output_path = os.path.join(OUTPUT_DIR, 'aggregate_model_reports.csv')
    df.to_csv(output_path, index=False)
    
    return df

def create_visualization_plots(df):
    """Create various visualization plots to compare model performances."""
    # Set up the plotting style
    sns.set_theme(style="whitegrid")  # This properly configures seaborn's styling
    
    # 1. Performance vs Computation Time
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['avg_execution_time'], 
                         df['prediction_accuracy'],
                         c=df['model_params_b'].astype(float),
                         s=200,
                         alpha=0.6,
                         cmap='viridis')
    plt.colorbar(scatter, label='Model Parameters (B)')
    plt.xlabel('Average Execution Time (s)')
    plt.ylabel('Prediction Accuracy (%)')
    plt.title('Model Performance vs Computation Time')
    
    # Add quantization annotations
    for idx, row in df.iterrows():
        plt.annotate(row['quantization'],
                    (row['avg_execution_time'], row['prediction_accuracy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'perf_vs_compute.png'))
    plt.close()
    
    # 2. Model Size Impact on Performance Metrics
    plt.figure(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1_score', 'auc_roc']
    
    for metric in metrics:
        sns.boxplot(x='model_params_b', y=metric, data=df)
    
    plt.xlabel('Model Parameters (B)')
    plt.ylabel('Score')
    plt.title('Impact of Model Size on Performance Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'size_impact.png'))
    plt.close()
    
    # 3. Quantization Impact on Loading Time
    plt.figure(figsize=(10, 6))
    sns.barplot(x='quantization', y='avg_load_duration', 
                hue='model_params_b', data=df)
    plt.xlabel('Quantization Type')
    plt.ylabel('Average Load Duration (ns)')
    plt.title('Impact of Quantization on Model Loading Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'quant_impact.png'))
    plt.close()
    
    # 4. Performance Tradeoffs
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['precision'], df['recall'],
                        c=df['avg_execution_time'],
                        s=df['model_params_b'].astype(float) * 20,
                        alpha=0.6,
                        cmap='coolwarm')
    
    plt.colorbar(scatter, label='Avg Execution Time (s)')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Tradeoff by Model Size and Speed')
    
    # Add model size legend
    sizes = sorted(df['model_params_b'].unique())
    legend_elements = [plt.scatter([], [], s=size*20, 
                                 c='gray', alpha=0.6,
                                 label=f'{size}B params')
                      for size in sizes]
    plt.legend(handles=legend_elements, title='Model Size',
              bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'perf_tradeoffs.png'))
    plt.close()

def main():
    """Main execution function."""
    print("Starting report aggregation...")
    df = aggregate_reports()
    
    if len(df) == 0:
        print("No reports were successfully processed. Please check the directory structure and file contents.")
        return
        
    print(f"Successfully processed {len(df)} reports.")
    print("\nDataFrame columns:")
    print(df.columns.tolist())
    print("\nFirst row preview:")
    print(df.iloc[0] if len(df) > 0 else "No data")
    
    print("\nCreating visualizations...")
    create_visualization_plots(df)
    print("Done! Output saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()