import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import logging
from datetime import datetime

# ------------------------------------------------------------------------------
# 1. Logging and Utility Functions
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_metric(value):
    """
    Normalize an accuracy or f1_score to the 0.0-1.0 range.
    If greater than 1 (e.g., 75), interpret as percentage and convert to 0.75.
    """
    try:
        if pd.isna(value):
            return np.nan
        
        value = float(value)
        if value > 1.0:
            value = value / 100.0
        return value
    except Exception as e:
        logger.warning(f"Error normalizing metric value {value}: {str(e)}")
        return np.nan

def calculate_auc_roc(row):
    """
    Compute or retrieve AUC from the row. 
    Customize this to compute from actual labels if you have them,
    or just return row['auc_roc'] if present.
    """
    try:
        # If there's an existing auc_roc column with a numeric value, just return it
        if 'auc_roc' in row and pd.notna(row['auc_roc']):
            return float(row['auc_roc'])
        
        # Example for real calculation:
        # if 'prediction_probabilities' in row and 'actual_labels' in row:
        #     return roc_auc_score(row['actual_labels'], row['prediction_probabilities'])
        
        return np.nan
    except Exception as e:
        logger.warning(f"Error computing AUC for row: {str(e)}")
        return np.nan

# ------------------------------------------------------------------------------
# 2. Define a Mapping for Agenticsim -> Standard LLM Model Names
#    Ensures both CSVs use the same model_name for merges.
# ------------------------------------------------------------------------------

rename_map_agenticsim_to_standard = {
    "aya_expanse_8b_q4_k_m":          "aya-expanse:8b-q4",
    "deepseek_r1_7b":                "deepseek-r1:7b",
    "dolphin3_8b_llama3_1_q4_k_m":   "dolphin3:8b-llama3.1-q4",
    "exaone3_5_7_8b_instruct_q4_k_m":"exaone3.5:7.8b-instruct-q4",
    "falcon3_7b_instruct_q4_k_m":    "falcon3:7b-instruct-q4",
    "gemma2_9b_instruct_q4_k_m":     "gemma2:9b-instruct-q4",
    "glm4_9b_chat_q4_k_m":           "glm4:9b-chat-q4",
    "granite3_1_dense_8b_instruct_q4_k_m": "granite3.1-dense:8b-instruct-q4",
    "hermes3_8b_llama3_1_q4_k_m":    "hermes3:8b-llama3.1-q4",
    "llama3_1_8b_instruct_q4_k_m":   "llama3.1:8b-instruct-q4",
    "marco_o1_7b_q4_k_m":            "marco-o1:7b-q4",
    "mistral_7b_instruct_q4_k_m":    "mistral:7b-instruct-q4",
    "olmo2_7b_1124_instruct_q4_k_m": "olmo2:7b-1124-instruct-q4",
    "phi4_14b_q4_k_m":               "phi4:14b-q4",
    "qwen2_5_7b_instruct_q4_k_m":    "qwen2.5:7b-instruct-q4",
    "tulu3_8b_q4_k_m":               "tulu3:8b-q4"
}

# ------------------------------------------------------------------------------
# 3. Column Mapping With Direct Python Callables
#    Replace "CONST=..." and "LAMBDA=..." with actual functions/lambdas.
# ------------------------------------------------------------------------------

def compute_predict_yes_ct(row):
    """Compute predict_yes_ct from agenticsim columns, with safe fallback."""
    # row.get("prediction_yes_percent", np.nan) could be NaN, so coerce to 0 if NaN
    yes_percent = row.get("prediction_yes_percent", np.nan)
    if pd.isna(yes_percent):
        yes_percent = 0
    
    calls = row.get("api_call_ct", 0)
    return int(yes_percent * calls / 100.0)

def compute_predict_no_ct(row):
    """Compute predict_no_ct = total calls - predict_yes_ct, for agenticsim rows."""
    calls = row.get("api_call_ct", 0)
    yes_ct = compute_predict_yes_ct(row)
    return max(0, calls - yes_ct)

def compute_api_prompt_eval_count(row):
    """
    Sum the speaker{i}_prompt_eval_ct_median columns for i in [1..5],
    ignoring NaNs.
    """
    total = 0
    for i in range(1, 6):
        val = row.get(f"speaker{i}_prompt_eval_ct_median", np.nan)
        if pd.notna(val):
            total += val
    return total

def compute_api_eval_count(row):
    """
    Sum the speaker{i}_eval_ct_median columns for i in [1..5],
    ignoring NaNs.
    """
    total = 0
    for i in range(1, 6):
        val = row.get(f"speaker{i}_eval_ct_median", np.nan)
        if pd.notna(val):
            total += val
    return total

column_map_dt = {
    'model_name': {
        'INPUT_A': lambda row: row.get('model_name', np.nan),
        'INPUT_B': lambda row: row.get('model_name', np.nan)
    },
    'prompt_type': {
        'INPUT_A': lambda row: row.get('prompt_type', np.nan),
        'INPUT_B': lambda row: "agenticsim"
    },
    'accuracy': {
        'INPUT_A': lambda row: normalize_metric(row.get('accuracy', np.nan)),
        'INPUT_B': lambda row: normalize_metric(row.get('accuracy', np.nan))
    },
    'f1_score': {
        'INPUT_A': lambda row: normalize_metric(row.get('f1_score', np.nan)),
        'INPUT_B': lambda row: normalize_metric(row.get('f1_score', np.nan))
    },
    'api_call_ct': {
        'INPUT_A': lambda row: row.get('api_call_total_ct', np.nan),
        'INPUT_B': lambda row: row.get('api_call_ct', np.nan)
    },
    'api_success_ct': {
        'INPUT_A': lambda row: row.get('api_call_success_ct', np.nan),
        'INPUT_B': lambda row: row.get('api_success_ct', np.nan)
    },
    'predict_yes_ct': {
        'INPUT_A': lambda row: row.get('actual_yes_ct', np.nan),
        'INPUT_B': compute_predict_yes_ct
    },
    'predict_no_ct': {
        'INPUT_A': lambda row: row.get('actual_no_ct', np.nan),
        'INPUT_B': compute_predict_no_ct
    },
    'true_positive': {
        'INPUT_A': lambda row: row.get('confusion_tp', np.nan),
        'INPUT_B': lambda row: row.get('true_positive', np.nan)
    },
    'true_negative': {
        'INPUT_A': lambda row: row.get('confusion_tn', np.nan),
        'INPUT_B': lambda row: row.get('true_negative', np.nan)
    },
    'false_positive': {
        'INPUT_A': lambda row: row.get('confusion_fp', np.nan),
        'INPUT_B': lambda row: row.get('false_positive', np.nan)
    },
    'false_negative': {
        'INPUT_A': lambda row: row.get('confusion_fn', np.nan),
        'INPUT_B': lambda row: row.get('false_negative', np.nan)
    },
    'auc_roc': {
        'INPUT_A': lambda row: calculate_auc_roc(row),
        'INPUT_B': lambda row: calculate_auc_roc(row)
    },
    'api_duration_sec': {
        'INPUT_A': lambda row: row.get('api_total_duration_sec', np.nan),
        'INPUT_B': lambda row: row.get('speaker_all_total_duration_sec_median', np.nan)
    },
    'api_prompt_eval_count': {
        'INPUT_A': lambda row: row.get('api_prompt_eval_count', np.nan),
        'INPUT_B': compute_api_prompt_eval_count
    },
    'api_eval_count': {
        'INPUT_A': lambda row: "",  # or np.nan, if you prefer
        'INPUT_B': compute_api_eval_count
    },
    'confidence_median': {
        'INPUT_A': lambda row: "",  # or np.nan
        'INPUT_B': lambda row: row.get('confidence_median', np.nan)
    }
}

# ------------------------------------------------------------------------------
# 4. Performance Report Generation
# ------------------------------------------------------------------------------

def calculate_statistics(series, metric_name):
    """
    Safely calculate mean, median, std, mode for a numeric series, ignoring NaN.
    """
    stats = {}
    try:
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            logger.warning(f"No valid data points for {metric_name} statistics")
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'mode': np.nan
            }
        
        stats['mean'] = clean_series.mean()
        stats['median'] = clean_series.median()
        stats['std'] = clean_series.std()
        
        # Mode can be multiple values; we just take the first if it exists
        mode_result = clean_series.mode()
        stats['mode'] = mode_result.iloc[0] if not mode_result.empty else np.nan
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating statistics for {metric_name}: {str(e)}")
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'mode': np.nan
        }

def generate_performance_report(df):
    """
    Generate a performance analysis report for models that have 'agenticsim' rows.
    Compares 'agenticsim' vs 'cot' vs 'cot-nshot'.
    """
    report_sections = []
    
    try:
        agenticsim_models = df[df['prompt_type'] == 'agenticsim'] \
            .sort_values(by='f1_score', ascending=False)['model_name'] \
            .unique()
        
        if len(agenticsim_models) == 0:
            logger.warning("No models found with 'agenticsim' prompt type.")
            return "No models found with 'agenticsim' prompt type for analysis."
        
        # Intro
        report_sections.extend([
            "1. Model Performance Rankings by Prompt Type",
            "=" * 50,
            f"\nAnalysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ])
        
        boost_data = []
        
        for model in agenticsim_models:
            model_df = df[df['model_name'] == model]
            try:
                agenticsim_rows = model_df[model_df['prompt_type'] == 'agenticsim']
                if agenticsim_rows.empty:
                    continue
                agenticsim_row = agenticsim_rows.iloc[0]
                
                report_sections.extend([
                    f"\n{model}",
                    "-" * len(model),
                    f"AgenticSim: F1={agenticsim_row['f1_score']:.3f}, Acc={agenticsim_row['accuracy']:.3f}"
                ])
                
                # Print other prompt types
                other_prompts = model_df[model_df['prompt_type'] != 'agenticsim'] \
                    .sort_values(by='f1_score', ascending=False)
                for _, row_prompt in other_prompts.iterrows():
                    report_sections.append(
                        f"{row_prompt['prompt_type']}: "
                        f"F1={row_prompt['f1_score']:.3f}, "
                        f"Acc={row_prompt['accuracy']:.3f}"
                    )
                
                # Compute boosts
                cot_boost = np.nan
                cotnshot_boost = np.nan
                
                cot_row = model_df[model_df['prompt_type'] == 'cot']
                if not cot_row.empty:
                    f1_cot = cot_row.iloc[0]['f1_score']
                    if pd.notna(f1_cot) and pd.notna(agenticsim_row['f1_score']):
                        cot_boost = agenticsim_row['f1_score'] - f1_cot
                    report_sections.append(
                        f"Agentic Reasoning Boost (CoT): "
                        f"{cot_boost if pd.notna(cot_boost) else 'NA'}"
                    )
                
                cotnshot_row = model_df[model_df['prompt_type'] == 'cot-nshot']
                if not cotnshot_row.empty:
                    f1_cotnshot = cotnshot_row.iloc[0]['f1_score']
                    if pd.notna(f1_cotnshot) and pd.notna(agenticsim_row['f1_score']):
                        cotnshot_boost = agenticsim_row['f1_score'] - f1_cotnshot
                    report_sections.append(
                        f"Agentic Reasoning Boost (CoT-NShot): "
                        f"{cotnshot_boost if pd.notna(cotnshot_boost) else 'NA'}"
                    )
                
                boost_data.append({
                    'model': model,
                    'cot_boost': cot_boost,
                    'cotnshot_boost': cotnshot_boost
                })
            
            except Exception as ex:
                logger.error(f"Error processing model {model}: {str(ex)}")
                continue
        
        # 2. Summarize Boost Rankings
        report_sections.extend([
            "\n\n2. Agentic Reasoning Boost Rankings",
            "=" * 50
        ])
        
        boost_df = pd.DataFrame(boost_data)
        
        # CoT Boost
        report_sections.extend([
            "\nCoT Boost Rankings:",
            "-" * 20
        ])
        if 'cot_boost' in boost_df.columns and not boost_df['cot_boost'].dropna().empty:
            cot_rankings = boost_df.dropna(subset=['cot_boost']) \
                                   .sort_values(by='cot_boost', ascending=False)
            for _, rowb in cot_rankings.iterrows():
                report_sections.append(f"{rowb['model']}: {rowb['cot_boost']:.3f}")
            
            cot_stats = calculate_statistics(boost_df['cot_boost'], 'CoT boost')
            report_sections.extend([
                "\nCoT Boost Statistics:",
                f"Mean: {cot_stats['mean']:.3f}",
                f"Median: {cot_stats['median']:.3f}",
                f"Std Dev: {cot_stats['std']:.3f}",
                f"Mode: {cot_stats['mode']:.3f}"
            ])
        else:
            report_sections.append("No valid CoT boost data.")
        
        # CoT-NShot Boost
        report_sections.extend([
            "\nCoT-NShot Boost Rankings:",
            "-" * 20
        ])
        if 'cotnshot_boost' in boost_df.columns and not boost_df['cotnshot_boost'].dropna().empty:
            cotnshot_rankings = boost_df.dropna(subset=['cotnshot_boost']) \
                                        .sort_values(by='cotnshot_boost', ascending=False)
            for _, rowb in cotnshot_rankings.iterrows():
                report_sections.append(f"{rowb['model']}: {rowb['cotnshot_boost']:.3f}")
            
            cotnshot_stats = calculate_statistics(boost_df['cotnshot_boost'], 'CoT-NShot boost')
            report_sections.extend([
                "\nCoT-NShot Boost Statistics:",
                f"Mean: {cotnshot_stats['mean']:.3f}",
                f"Median: {cotnshot_stats['median']:.3f}",
                f"Std Dev: {cotnshot_stats['std']:.3f}",
                f"Mode: {cotnshot_stats['mode']:.3f}"
            ])
        else:
            report_sections.append("No valid CoT-NShot boost data.")
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        return f"Error generating performance report: {str(e)}"
    
    return "\n".join(report_sections)

# ------------------------------------------------------------------------------
# 5. Main Script: Reading Data, Applying Transformations, Saving, Reporting
# ------------------------------------------------------------------------------

def transform_row(row, input_key):
    """
    Transform a single row from either the standard LLM CSV or the agenticsim CSV
    based on column_map_dt.
    """
    new_row = {}
    for col_name, mapping in column_map_dt.items():
        if input_key in mapping:
            func = mapping[input_key]
            try:
                val = func(row)
                new_row[col_name] = val
            except Exception as e:
                logger.error(f"Error processing column '{col_name}' from row {row} - {e}")
                new_row[col_name] = np.nan
        else:
            # No mapping defined for this input type
            new_row[col_name] = np.nan
    return new_row

def main():
    # Adjust paths to suit your file locations
    INPUT_STANDARDLLM_FILEPATH = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                             'aggregation_summary_report.csv')
    INPUT_AGENTICSIMLLM_FILEPATH = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                               'transcripts_stat_summary_final_20250128_013836.csv')
    OUTPUT_FILEPATH = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                   'combined_evaluations-transcripts_manual.csv')
    
    logger.info("Starting data processing")
    logger.info(f"Reading standard LLM data from: {INPUT_STANDARDLLM_FILEPATH}")
    logger.info(f"Reading agenticsim LLM data from: {INPUT_AGENTICSIMLLM_FILEPATH}")
    
    try:
        standardllm_df = pd.read_csv(INPUT_STANDARDLLM_FILEPATH)
        logger.info(f"Successfully read {len(standardllm_df)} rows from standard LLM file")
    except Exception as e:
        logger.error(f"Error reading standard LLM file: {str(e)}")
        return
    
    try:
        agenticsimllm_df = pd.read_csv(INPUT_AGENTICSIMLLM_FILEPATH)
        logger.info(f"Successfully read {len(agenticsimllm_df)} rows from agenticsim LLM file")
    except Exception as e:
        logger.error(f"Error reading agenticsim LLM file: {str(e)}")
        return
    
    # Rename agenticsim model names to match standard
    agenticsimllm_df['model_name'] = agenticsimllm_df['model_name'].replace(rename_map_agenticsim_to_standard)
    
    # Create containers for final rows
    logger.info(f"Created empty DataFrame with columns: {', '.join(column_map_dt.keys())}")
    input_a_data = []
    input_b_data = []
    
    # Process (INPUT_A): Standard LLM
    logger.info("Processing standard LLM data...")
    for _, row in standardllm_df.iterrows():
        new_row = transform_row(row, 'INPUT_A')
        input_a_data.append(new_row)
    logger.info(f"Processed {len(input_a_data)} rows from standard LLM data")
    
    # Process (INPUT_B): Agenticsim LLM
    logger.info("Processing agenticsim LLM data...")
    for _, row in agenticsimllm_df.iterrows():
        new_row = transform_row(row, 'INPUT_B')
        input_b_data.append(new_row)
    logger.info(f"Processed {len(input_b_data)} rows from agenticsim LLM data")
    
    # Combine them
    combined_df = pd.DataFrame(input_a_data + input_b_data)
    logger.info(f"Combined data created with {len(combined_df)} total rows")
    
    # Save combined dataset
    try:
        combined_df.to_csv(OUTPUT_FILEPATH, index=False)
        logger.info(f"Combined data saved to: {OUTPUT_FILEPATH}")
    except Exception as e:
        logger.error(f"Error saving combined data: {str(e)}")
    
    # Show sample
    print("\nSample of combined data:")
    print(combined_df.head())
    
    # Generate and print performance report
    logger.info("Generating performance analysis report...")
    report = generate_performance_report(combined_df)
    
    # Save report
    report_filepath = os.path.join(
        '..', 'aggregation_evaluations-transcripts_manual-copy',
        'performance_analysis_report.txt'
    )
    try:
        with open(report_filepath, 'w') as f:
            f.write(report)
        logger.info(f"Performance report saved to: {report_filepath}")
    except Exception as e:
        logger.error(f"Error saving performance report: {str(e)}")
    
    # Print final report
    print("\nPerformance Analysis Report")
    print("=" * 80)
    print(report)
    
    # Print data quality stats
    print("\nData Quality Statistics:")
    print("-" * 20)
    for col in combined_df.columns:
        non_null_count = combined_df[col].count()
        completeness = (non_null_count / len(combined_df)) * 100
        print(f"{col}: {completeness:.1f}% complete ({non_null_count}/{len(combined_df)} rows)")

if __name__ == "__main__":
    main()
