#!/usr/bin/env python3
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# 0. Global constants
EVALUATION_RESULTS_DIR = 'evaluation_results_long_20250110'
ROOT_DIR = os.path.abspath(EVALUATION_RESULTS_DIR)  # absolute path
OUTPUT_DIR = os.path.join('data', 'aggregate_reports', os.path.basename(EVALUATION_RESULTS_DIR))

if not os.path.exists(ROOT_DIR):
    raise FileNotFoundError(f"Evaluation directory not found: {ROOT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_model_metadata(model_dir_name):
    """
    Extract model size and quantization information from directory name,
    e.g. 'aya-expanse_8b-q4_k_m' => param: '8', quant: 'q4_k_m'.
    """
    # e.g. re.search(r'(\d+)b', 'aya-expanse_8b-q4_k_m') => '8'
    param_match = re.search(r'(\d+)b', model_dir_name.lower())
    quant_match = re.search(r'(q4_k_m|fp16|q4_k_m)', model_dir_name.lower())

    params = param_match.group(1) if param_match else "unknown"
    quant = quant_match.group(1) if quant_match else "unknown"

    return {
        'model_params_b': params,
        'quantization': quant
    }


def parse_report_json(json_path):
    """
    Parse a JSON report file and return a dictionary of relevant fields.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create a flat dictionary with top-level metrics
    stats = {
        'model_name': data.get('model_name'),
        'prompt_type': data.get('prompt_type'),
        'total_attempts': data.get('total_attempts'),
        'successful_attempts': data.get('successful_attempts'),
        'failed_attempts': data.get('failed_attempts'),
        'timeout_attempts': data.get('timeout_attempts'),
        'avg_execution_time': data.get('avg_execution_time'),
        'median_execution_time': data.get('median_execution_time'),
        'sd_execution_time': data.get('sd_execution_time'),
        'prediction_accuracy': data.get('prediction_accuracy'),
        'auc_roc': data.get('auc_roc'),
    }

    # metadata averages
    meta_data_averages = data.get('meta_data_averages', {})
    stats['avg_total_duration'] = meta_data_averages.get('total_duration')
    stats['avg_load_duration'] = meta_data_averages.get('load_duration')
    stats['avg_prompt_eval_duration'] = meta_data_averages.get('prompt_eval_duration')
    stats['avg_eval_duration'] = meta_data_averages.get('eval_duration')

    # confusion matrix
    confusion = data.get('confusion_matrix', {})
    stats['true_positives'] = confusion.get('tp')
    stats['true_negatives'] = confusion.get('tn')
    stats['false_positives'] = confusion.get('fp')
    stats['false_negatives'] = confusion.get('fn')

    # Add derived metrics (precision, recall, f1) if possible
    tp = stats['true_positives'] or 0
    fp = stats['false_positives'] or 0
    fn = stats['false_negatives'] or 0

    # Avoid zero-division
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    stats['precision'] = precision
    stats['recall'] = recall
    stats['f1_score'] = f1_score

    return stats


def parse_report_txt(txt_path):
    """
    Parse a TXT file containing summary statistics (like the sample text).
    Return a dictionary of data. You can customize to parse more lines.
    """
    summary = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line_stripped = line.strip()
        # Example parse: 'Accuracy: 53.00%' => 53.0
        if line_stripped.startswith("Accuracy:"):
            val = line_stripped.split("Accuracy:")[-1].strip()
            val = val.replace('%', '')
            try:
                summary['txt_accuracy'] = float(val)
            except ValueError:
                summary['txt_accuracy'] = None

        # Example parse: 'AUC-ROC: 0.4699' => 0.4699
        if "AUC-ROC:" in line_stripped:
            val = line_stripped.split("AUC-ROC:")[-1].strip()
            try:
                summary['txt_auc_roc'] = float(val)
            except ValueError:
                summary['txt_auc_roc'] = None

        # Similarly parse other lines if needed:
        # e.g., total duration, median execution, etc.

    return summary


def process_report_files(json_path, txt_path):
    """
    Process a JSON report and its corresponding TXT file.
    Return a combined dictionary of all metrics we care about.
    """
    # Parse JSON
    try:
        json_stats = parse_report_json(json_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON file: {json_path} => {e}")
        return None

    # Parse TXT
    try:
        txt_stats = parse_report_txt(txt_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse TXT file: {txt_path} => {e}")
        txt_stats = {}

    # Combine
    combined_stats = {**json_stats, **txt_stats}
    return combined_stats


def aggregate_reports():
    """
    Crawl the directory tree and aggregate all .json + .txt
    pairs found in 'reports' directories two levels down from ROOT_DIR.
    """
    print(f"[INFO] Starting directory crawl from: {ROOT_DIR}")
    aggregated_rows = []

    # Full walk
    for root, dirs, files in os.walk(ROOT_DIR):
        if os.path.basename(root) == 'reports':
            # The parent of 'reports' is presumably the model directory
            model_dir_name = os.path.basename(os.path.dirname(root))
            model_meta = extract_model_metadata(model_dir_name)

            json_files = [f for f in files if f.endswith('.json')]
            for jf in json_files:
                base_name = os.path.splitext(jf)[0]
                tf = base_name + '.txt'
                if tf in files:
                    json_path = os.path.join(root, jf)
                    txt_path = os.path.join(root, tf)

                    # Process files
                    combined_stats = process_report_files(json_path, txt_path)
                    if combined_stats is not None:
                        # Add model metadata
                        combined_stats.update(model_meta)
                        # Also store which directory we came from
                        combined_stats['model_dir'] = model_dir_name
                        aggregated_rows.append(combined_stats)
                        print(f"[INFO] Added report data for: {jf}")

    # Convert to DataFrame
    df = pd.DataFrame(aggregated_rows)
    if len(df) == 0:
        print("[WARNING] No valid reports were found.")
    return df


def create_visualization_plots(df):
    """Create some example plots to compare model performances."""
    if df.empty:
        print("[INFO] DataFrame is empty; skipping plots.")
        return

    sns.set_theme(style="whitegrid")

    # 1. Performance vs Computation Time
    plt.figure(figsize=(12, 8))
    # For color mapping, we need numeric values:
    # e.g., convert string 'model_params_b' to float if possible
    df['model_params_b_float'] = pd.to_numeric(df['model_params_b'], errors='coerce')
    scatter = plt.scatter(df['avg_execution_time'],
                          df['prediction_accuracy'],
                          c=df['model_params_b_float'],
                          s=200, alpha=0.7, cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Model Parameters (B)', rotation=270, labelpad=15)
    plt.xlabel('Average Execution Time (s)')
    plt.ylabel('Prediction Accuracy (%)')
    plt.title('Model Performance vs. Computation Time')

    # Annotate points with quantization
    for idx, row in df.iterrows():
        plt.annotate(row['quantization'],
                     (row['avg_execution_time'], row['prediction_accuracy']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plot_name = os.path.join(OUTPUT_DIR, 'perf_vs_compute.png')
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {plot_name}")

    # 2. Model Size Impact on Performance Metrics (boxplot)
    plt.figure(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1_score', 'auc_roc']
    # Melt the dataframe for seaborn boxplot across multiple metrics
    plot_df = df.melt(id_vars=['model_params_b'], value_vars=metrics,
                      var_name='metric', value_name='score')
    sns.boxplot(x='model_params_b', y='score', hue='metric', data=plot_df)
    plt.xlabel('Model Parameters (B)')
    plt.ylabel('Score')
    plt.title('Impact of Model Size on Performance Metrics')
    plt.legend(loc='best')
    plt.tight_layout()

    plot_name = os.path.join(OUTPUT_DIR, 'size_impact.png')
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {plot_name}")

    # 3. Quantization Impact on Loading Time (barplot)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='quantization', y='avg_load_duration',
                hue='model_params_b', data=df)
    plt.xlabel('Quantization Type')
    plt.ylabel('Average Load Duration (ns)')
    plt.title('Impact of Quantization on Model Loading Time')
    plt.tight_layout()

    plot_name = os.path.join(OUTPUT_DIR, 'quant_impact.png')
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {plot_name}")

    # 4. Precision-Recall Tradeoff
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['precision'], df['recall'],
                          c=df['avg_execution_time'],
                          s=df['model_params_b_float'] * 20, alpha=0.6,
                          cmap='coolwarm')
    plt.colorbar(scatter, label='Avg Execution Time (s)')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Tradeoff by Model Size & Speed')

    # Optional: add legend for model size
    unique_sizes = sorted(df['model_params_b_float'].dropna().unique())
    legend_elements = [
        plt.scatter([], [], s=size*20, c='gray', alpha=0.6,
                    label=f'{int(size)}B') for size in unique_sizes
    ]
    plt.legend(handles=legend_elements, title='Model Size (B)',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plot_name = os.path.join(OUTPUT_DIR, 'perf_tradeoffs.png')
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {plot_name}")


def main():
    print("[INFO] Starting report aggregation...")
    df = aggregate_reports()

    # If we have some data, proceed with saving and plotting
    if not df.empty:
        # Save aggregated CSV
        output_csv = os.path.join(OUTPUT_DIR, 'aggregate_model_reports.csv')
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Aggregated {len(df)} rows saved to: {output_csv}")

        # Create plots
        create_visualization_plots(df)
        print(f"[INFO] Visualization plots saved to: {OUTPUT_DIR}")
    else:
        print("[WARNING] No data to save or plot. Exiting.")


if __name__ == "__main__":
    main()
