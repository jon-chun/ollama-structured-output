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
        'INPUT_B': 'LAMBDA=int(row.prediction_yes_percent*row.api_call_ct/100)'
    },
    'predict_no_ct': {
        'INPUT_A': 'actual_no_ct',
        'INPUT_B': 'LAMBDA=int(row.api_call_ct-int(row.prediction_yes_percent*row.api_call_ct/100))'
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
        'INPUT_B': 'LAMBDA=sum([row[f"speaker{i}_prompt_eval_ct_median"] for i in range(1,6) if pd.notna(row[f"speaker{i}_prompt_eval_ct_median"])])'
    },
    'api_eval_count': {
        'INPUT_A': 'CONST=""',
        'INPUT_B': 'LAMBDA=sum([row[f"speaker{i}_eval_ct_median"] for i in range(1,6) if pd.notna(row[f"speaker{i}_eval_ct_median"])])'
    },
    'confidence_median': {
        'INPUT_A': 'CONST=""',
        'INPUT_B': 'confidence_median'
    }
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
        if value > 1:
            value = value / 100.0
        logger.debug(f"Normalized metric value from {value} to {value:.3f}")
        return value
    except Exception as e:
        logger.warning(f"Error normalizing metric value {value}: {str(e)}")
        return np.nan

def calculate_statistics(series, metric_name):
    """
    Safely calculate statistics for a series, handling empty or all-NaN cases
    """
    stats = {}
    try:
        # Remove NaN values for calculations
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
        
        logger.debug(f"Statistics calculated for {metric_name}: {stats}")
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
                # Get agenticsim scores
                agenticsim_row = model_df[model_df['prompt_type'] == 'agenticsim'].iloc[0]
                report_sections.extend([
                    f"\n{model}",
                    "-" * len(model),
                    f"AgenticSim: F1={agenticsim_row['f1_score']:.3f}, Acc={agenticsim_row['accuracy']:.3f}"
                ])
                
                # Get other prompt types
                other_prompts = model_df[model_df['prompt_type'] != 'agenticsim'].sort_values(
                    by='f1_score', ascending=False)
                
                for _, row in other_prompts.iterrows():
                    report_sections.append(
                        f"{row['prompt_type']}: F1={row['f1_score']:.3f}, Acc={row['accuracy']:.3f}")
                
                # Calculate boosts
                cot_boost = np.nan
                cotnshot_boost = np.nan
                
                cot_row = model_df[model_df['prompt_type'] == 'cot']
                if not cot_row.empty:
                    cot_boost = agenticsim_row['f1_score'] - cot_row.iloc[0]['f1_score']
                    report_sections.append(f"Agentic Reasoning Boost (CoT): {cot_boost:.3f}")
                
                cotnshot_row = model_df[model_df['prompt_type'] == 'cot-nshot']
                if not cotnshot_row.empty:
                    cotnshot_boost = agenticsim_row['f1_score'] - cotnshot_row.iloc[0]['f1_score']
                    report_sections.append(f"Agentic Reasoning Boost (CoT-NShot): {cotnshot_boost:.3f}")
                
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
        
        cot_rankings = boost_df.sort_values(by='cot_boost', ascending=False)
        for _, row in cot_rankings.iterrows():
            if not pd.isna(row['cot_boost']):
                report_sections.append(f"{row['model']}: {row['cot_boost']:.3f}")
        
        # Calculate CoT boost statistics
        cot_stats = calculate_statistics(boost_df['cot_boost'], 'CoT boost')
        report_sections.extend([
            "\nCoT Boost Statistics:",
            f"Mean: {cot_stats['mean']:.3f}",
            f"Median: {cot_stats['median']:.3f}",
            f"Std Dev: {cot_stats['std']:.3f}",
            f"Mode: {cot_stats['mode']:.3f}"
        ])
        
        # CoT-NShot Boost Rankings
        report_sections.extend([
            "\nCoT-NShot Boost Rankings:",
            "-" * 20
        ])
        
        cotnshot_rankings = boost_df.sort_values(by='cotnshot_boost', ascending=False)
        for _, row in cotnshot_rankings.iterrows():
            if not pd.isna(row['cotnshot_boost']):
                report_sections.append(f"{row['model']}: {row['cotnshot_boost']:.3f}")
        
        # Calculate CoT-NShot boost statistics
        cotnshot_stats = calculate_statistics(boost_df['cotnshot_boost'], 'CoT-NShot boost')
        report_sections.extend([
            "\nCoT-NShot Boost Statistics:",
            f"Mean: {cotnshot_stats['mean']:.3f}",
            f"Median: {cotnshot_stats['median']:.3f}",
            f"Std Dev: {cotnshot_stats['std']:.3f}",
            f"Mode: {cotnshot_stats['mode']:.3f}"
        ])
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        return f"Error generating performance report: {str(e)}"
    
    return "\n".join(report_sections)

def main():
    # Step 1: Read input files
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
    
    # Create empty DataFrame with desired columns
    combined_df = pd.DataFrame(columns=column_map_dt.keys())
    logger.info(f"Created empty DataFrame with columns: {', '.join(column_map_dt.keys())}")
    
    # Process INPUT_A (standardllm_df)
    input_a_data = []
    logger.info("Processing standard LLM data...")
    
    for _, row in standardllm_df.iterrows():
        new_row = {}
        for col_name, mapping in column_map_dt.items():
            if 'INPUT_A' in mapping:
                value = mapping['INPUT_A']
                try:
                    if value.startswith('LAMBDA='):
                        lambda_expr = value.split('=', 1)[1]
                        new_row[col_name] = execute_lambda_function(pd.DataFrame([row]), lambda_expr).iloc[0]
                    elif value.startswith('CONST='):
                        const_value = value.split('=', 1)[1].strip('"')
                        new_row[col_name] = const_value
                    else:
                        if col_name in ['accuracy', 'f1_score']:
                            new_row[col_name] = normalize_metric(row[value])
                        else:
                            new_row[col_name] = row[value]
                except Exception as e:
                    logger.error(f"Error processing column {col_name} for standard LLM: {str(e)}")
                    new_row[col_name] = np.nan
        input_a_data.append(new_row)
    
    logger.info(f"Processed {len(input_a_data)} rows from standard LLM data")
    
    # Process INPUT_B (agenticsimllm_df)
    input_b_data = []
    logger.info("Processing agenticsim LLM data...")
    
    for _, row in agenticsimllm_df.iterrows():
        new_row = {}
        for col_name, mapping in column_map_dt.items():
            if 'INPUT_B' in mapping:
                value = mapping['INPUT_B']
                try:
                    if value.startswith('LAMBDA='):
                        lambda_expr = value.split('=', 1)[1]
                        new_row[col_name] = execute_lambda_function(pd.DataFrame([row]), lambda_expr).iloc[0]
                    elif value.startswith('CONST='):
                        const_value = value.split('=', 1)[1].strip('"')
                        new_row[col_name] = const_value
                    else:
                        if col_name in ['accuracy', 'f1_score']:
                            new_row[col_name] = normalize_metric(row[value])
                        else:
                            new_row[col_name] = row[value]
                except Exception as e:
                    logger.error(f"Error processing column {col_name} for agenticsim LLM: {str(e)}")
                    new_row[col_name] = np.nan
        input_b_data.append(new_row)
    
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
    report_filepath = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                 'performance_analysis_report.txt')
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