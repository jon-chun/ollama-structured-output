import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import logging
from datetime import datetime

# Define column mapping dictionary for data transformation
column_map_dt = {
    'model_name': {
        'INPUT_A': 'model_name',
        'INPUT_B': 'model_name'
    },
    'prompt_type': {
        'INPUT_A': 'prompt_type',
        'INPUT_B': 'CONST="agenticsim"'
    },
    'accuracy': {
        'INPUT_A': 'accuracy',
        'INPUT_B': 'accuracy'
    },
    'f1_score': {
        'INPUT_A': 'f1_score',
        'INPUT_B': 'f1_score'
    },
    'api_call_ct': {
        'INPUT_A': 'api_call_total_ct',
        'INPUT_B': 'api_call_ct'
    },
    'api_success_ct': {
        'INPUT_A': 'api_call_success_ct',
        'INPUT_B': 'api_success_ct'
    },
    'predict_yes_ct': {
        'INPUT_A': 'actual_yes_ct',
        'INPUT_B': 'LAMBDA=int(row.get("prediction_yes_percent", np.nan) * row.get("api_call_ct", 0) / 100)'
    },
    'predict_no_ct': {
        'INPUT_A': 'actual_no_ct',
        'INPUT_B': 'LAMBDA=int(row.get("api_call_ct", 0) - int(row.get("prediction_yes_percent", np.nan) \
                               * row.get("api_call_ct", 0) / 100))'
    },
    'true_positive': {
        'INPUT_A': 'confusion_tp',
        'INPUT_B': 'true_positive'
    },
    'true_negative': {
        'INPUT_A': 'confusion_tn',
        'INPUT_B': 'true_negative'
    },
    'false_positive': {
        'INPUT_A': 'confusion_fp',
        'INPUT_B': 'false_positive'
    },
    'false_negative': {
        'INPUT_A': 'confusion_fn',
        'INPUT_B': 'false_negative'
    },
    'auc_roc': {
        'INPUT_A': 'auc_roc',
        'INPUT_B': 'LAMBDA=calculate_auc_roc(row)'
    },
    'api_duration_sec': {
        'INPUT_A': 'api_total_duration_sec',
        'INPUT_B': 'speaker_all_total_duration_sec_median'
    },
    'api_prompt_eval_count': {
        'INPUT_A': 'api_prompt_eval_count',
        'INPUT_B': 'LAMBDA=sum([row.get(f"speaker{i}_prompt_eval_ct_median", np.nan) \
                                 for i in range(1,6) \
                                 if pd.notna(row.get(f"speaker{i}_prompt_eval_ct_median", np.nan))])'
    },
    'api_eval_count': {
        'INPUT_A': 'CONST=""',
        'INPUT_B': 'LAMBDA=sum([row.get(f"speaker{i}_eval_ct_median", np.nan) \
                                for i in range(1,6) \
                                if pd.notna(row.get(f"speaker{i}_eval_ct_median", np.nan))])'
    },
    'confidence_median': {
        'INPUT_A': 'CONST=""',
        'INPUT_B': 'confidence_median'
    }
}

# Agenticsim CSV --> Standard CSV
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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_metric(value):
    """
    Normalize accuracy or f1_score to 0.0-1.0 range
    """
    try:
        if pd.isna(value):
            return np.nan
        
        value = float(value)
        # If it's greater than 1 (e.g., 80 = 80%), convert to fraction
        if value > 1.0:
            value = value / 100.0
        return value
    except Exception as e:
        logger.warning(f"Error normalizing metric value {value}: {str(e)}")
        return np.nan

def calculate_auc_roc(row):
    """
    Compute AUC if the columns exist and are valid.
    Otherwise, return the row['auc_roc'] if it exists or np.nan.
    Modify to suit your data context.
    """
    try:
        # If there's already an auc_roc column with a numeric value, we can just return it
        # or properly compute from row['prediction_probabilities'] if desired.
        if 'auc_roc' in row and not pd.isna(row['auc_roc']):
            return float(row['auc_roc'])
        
        # Example: If we wanted to compute from raw data:
        # if 'prediction_probabilities' in row and 'actual_labels' in row:
        #     return roc_auc_score(row['actual_labels'], row['prediction_probabilities'])
        
        # Default fallback
        return np.nan
    except Exception as e:
        logger.warning(f"Error computing AUC for row: {str(e)}")
        return np.nan

def execute_lambda_function(row, expr_str):
    try:
        safe_locals = {
            # Minimal built-ins
            'int': int,
            'float': float,
            'sum': sum,
            'min': min,
            'max': max,
            'range': range,
            'pd': pd,
            
            # Your data references
            'row': row,
            'np': np,
            'pd': pd,
            'logger': logger,
            'calculate_auc_roc': calculate_auc_roc,
        }
        
        # Then call eval with an empty global environment plus the safe_locals as the local environment:
        return eval(expr_str, {"__builtins__": {}}, safe_locals)
    except Exception as e:
        logger.error(f"Failed to execute lambda expression '{expr_str}' with error: {str(e)}")
        return np.nan


def calculate_statistics(series, metric_name):
    """
    Safely calculate statistics for a series, handling empty or all-NaN cases
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
        
        # Safe mode calculation
        mode_result = clean_series.mode()
        stats['mode'] = mode_result.iloc[0] if len(mode_result) > 0 else np.nan
        
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
    Generate a comprehensive performance analysis report with error handling
    """
    report_sections = []
    
    try:
        # Get unique model names with agenticsim results
        agenticsim_models = df[df['prompt_type'] == 'agenticsim'].sort_values(
            by='f1_score', ascending=False)['model_name'].unique()
        
        if len(agenticsim_models) == 0:
            logger.warning("No models found with 'agenticsim' prompt type")
            return "No models found with 'agenticsim' prompt type for analysis"
        
        # Section 1: Model Rankings by Prompt Type
        report_sections.extend([
            "1. Model Performance Rankings by Prompt Type",
            "=" * 50,
            f"\nAnalysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ])
        
        boost_data = []
        
        for model in agenticsim_models:
            model_df = df[df['model_name'] == model]
            
            try:
                # Filter to get agenticsim row for this model
                agenticsim_row = model_df[model_df['prompt_type'] == 'agenticsim']
                if agenticsim_row.empty:
                    # Skip if no agenticsim row for this model
                    continue
                agenticsim_row = agenticsim_row.iloc[0]
                
                report_sections.extend([
                    f"\n{model}",
                    "-" * len(model),
                    f"AgenticSim: F1={agenticsim_row['f1_score']:.3f}, Acc={agenticsim_row['accuracy']:.3f}"
                ])
                
                # Get other prompt types
                other_prompts = model_df[model_df['prompt_type'] != 'agenticsim'].sort_values(
                    by='f1_score', ascending=False)
                
                for _, rowx in other_prompts.iterrows():
                    report_sections.append(
                        f"{rowx['prompt_type']}: F1={rowx['f1_score']:.3f}, Acc={rowx['accuracy']:.3f}"
                    )
                
                # Calculate boosts
                cot_boost = np.nan
                cotnshot_boost = np.nan
                
                # CoT
                cot_row = model_df[model_df['prompt_type'] == 'cot']
                if not cot_row.empty:
                    f1_cot = cot_row.iloc[0]['f1_score']
                    if pd.notna(f1_cot) and pd.notna(agenticsim_row['f1_score']):
                        cot_boost = agenticsim_row['f1_score'] - f1_cot
                    report_sections.append(f"Agentic Reasoning Boost (CoT): {cot_boost if pd.notna(cot_boost) else 'NA'}")
                
                # CoT-NShot
                cotnshot_row = model_df[model_df['prompt_type'] == 'cot-nshot']
                if not cotnshot_row.empty:
                    f1_cotnshot = cotnshot_row.iloc[0]['f1_score']
                    if pd.notna(f1_cotnshot) and pd.notna(agenticsim_row['f1_score']):
                        cotnshot_boost = agenticsim_row['f1_score'] - f1_cotnshot
                    report_sections.append(f"Agentic Reasoning Boost (CoT-NShot): {cotnshot_boost if pd.notna(cotnshot_boost) else 'NA'}")
                
                boost_data.append({
                    'model': model,
                    'cot_boost': cot_boost,
                    'cotnshot_boost': cotnshot_boost
                })
            
            except Exception as e:
                logger.error(f"Error processing model {model}: {str(e)}")
                continue
        
        # Section 2: Agentic Reasoning Boost Rankings
        report_sections.extend([
            "\n\n2. Agentic Reasoning Boost Rankings",
            "=" * 50
        ])
        
        boost_df = pd.DataFrame(boost_data)
        
        # CoT Boost Rankings
        report_sections.extend([
            "\nCoT Boost Rankings:",
            "-" * 20
        ])
        
        if 'cot_boost' in boost_df.columns and not boost_df['cot_boost'].dropna().empty:
            cot_rankings = boost_df.dropna(subset=['cot_boost']).sort_values(by='cot_boost', ascending=False)
            for _, rowb in cot_rankings.iterrows():
                report_sections.append(f"{rowb['model']}: {rowb['cot_boost']:.3f}")
            
            # Calculate CoT boost statistics
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
        
        # CoT-NShot Boost Rankings
        report_sections.extend([
            "\nCoT-NShot Boost Rankings:",
            "-" * 20
        ])
        
        if 'cotnshot_boost' in boost_df.columns and not boost_df['cotnshot_boost'].dropna().empty:
            cotnshot_rankings = boost_df.dropna(subset=['cotnshot_boost']).sort_values(by='cotnshot_boost', ascending=False)
            for _, rowb in cotnshot_rankings.iterrows():
                report_sections.append(f"{rowb['model']}: {rowb['cotnshot_boost']:.3f}")
            
            # Calculate CoT-NShot boost statistics
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

def main():
    INPUT_STANDARDLLM_FILEPATH = os.path.join(
        '..', 'aggregation_evaluations-transcripts_manual-copy', 'aggregation_summary_report.csv'
    )
    INPUT_AGENTICSIMLLM_FILEPATH = os.path.join(
        '..', 'aggregation_evaluations-transcripts_manual-copy', 'transcripts_stat_summary_final_20250128_013836.csv'
    )
    OUTPUT_FILEPATH = os.path.join(
        '..', 'aggregation_evaluations-transcripts_manual-copy', 'combined_evaluations-transcripts_manual.csv'
    )
    
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

    # Rename the agenticsim model_name to match the standard
    agenticsimllm_df['model_name'] = agenticsimllm_df['model_name'].replace(rename_map_agenticsim_to_standard)

    # Create empty DataFrame with desired columns
    combined_df = pd.DataFrame(columns=column_map_dt.keys())
    logger.info(f"Created empty DataFrame with columns: {', '.join(column_map_dt.keys())}")
    
    # Helper function to process one row according to the mapping
    def process_row(input_row, mapping_dict):
        new_row = {}
        for col_name, mapping in mapping_dict.items():
            if 'INPUT_A' in mapping:
                # We skip if we're not processing for input A in this pass
                continue
            if 'INPUT_B' in mapping:
                # We skip if we're not processing for input B in this pass
                continue
        return new_row

    # More generic approach: pass which input key to process
    def transform_row(row, input_key):
        """
        input_key: 'INPUT_A' or 'INPUT_B'
        row: a Series from the respective df
        """
        new_row = {}
        for col_name, mapping in column_map_dt.items():
            if input_key in mapping:
                expr = mapping[input_key]
                try:
                    # Check if it is a LAMBDA
                    if expr.startswith("LAMBDA="):
                        lambda_expr = expr.split("=", 1)[1]
                        new_row[col_name] = execute_lambda_function(row, lambda_expr)
                    elif expr.startswith("CONST="):
                        const_value = expr.split("=", 1)[1].strip('"')
                        new_row[col_name] = const_value
                    else:
                        # Normal column referencing
                        val = row.get(expr, np.nan)
                        if col_name in ['accuracy', 'f1_score']:
                            new_row[col_name] = normalize_metric(val)
                        else:
                            new_row[col_name] = val
                except Exception as e:
                    logger.error(f"Error processing column {col_name} for {input_key}: {str(e)}")
                    new_row[col_name] = np.nan
        return new_row
    
    # Process INPUT_A (standardllm_df)
    logger.info("Processing standard LLM data...")
    input_a_data = []
    for _, row in standardllm_df.iterrows():
        transformed = transform_row(row, 'INPUT_A')
        input_a_data.append(transformed)
    logger.info(f"Processed {len(input_a_data)} rows from standard LLM data")
    
    # Process INPUT_B (agenticsimllm_df)
    logger.info("Processing agenticsim LLM data...")
    input_b_data = []
    for _, row in agenticsimllm_df.iterrows():
        transformed = transform_row(row, 'INPUT_B')
        input_b_data.append(transformed)
    logger.info(f"Processed {len(input_b_data)} rows from agenticsim LLM data")
    
    # Combine processed data
    combined_df = pd.concat([
        pd.DataFrame(input_a_data),
        pd.DataFrame(input_b_data)
    ], ignore_index=True)
    
    logger.info(f"Combined data created with {len(combined_df)} total rows")
    
    # Save combined dataset
    try:
        combined_df.to_csv(OUTPUT_FILEPATH, index=False)
        logger.info(f"Combined data saved to: {OUTPUT_FILEPATH}")
    except Exception as e:
        logger.error(f"Error saving combined data: {str(e)}")
    
    print("\nSample of combined data:")
    print(combined_df.head())
    
    # Generate and print performance report
    logger.info("Generating performance analysis report...")
    report = generate_performance_report(combined_df)
    
    # Save report to file
    report_filepath = os.path.join(
        '..', 'aggregation_evaluations-transcripts_manual-copy', 'performance_analysis_report.txt'
    )
    try:
        with open(report_filepath, 'w') as f:
            f.write(report)
        logger.info(f"Performance report saved to: {report_filepath}")
    except Exception as e:
        logger.error(f"Error saving performance report: {str(e)}")
    
    print("\nPerformance Analysis Report")
    print("=" * 80)
    print(report)
    
    # Print data quality statistics
    print("\nData Quality Statistics:")
    print("-" * 20)
    for col in combined_df.columns:
        non_null_count = combined_df[col].count()
        completeness = (non_null_count / len(combined_df)) * 100
        print(f"{col}: {completeness:.1f}% complete ({non_null_count}/{len(combined_df)} rows)")

if __name__ == "__main__":
    main()
