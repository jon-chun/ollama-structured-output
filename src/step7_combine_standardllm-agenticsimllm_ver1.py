import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc_roc(row):
    """
    Calculate AUC-ROC from confusion matrix values
    """
    try:
        y_true = ([1] * int(row['true_positive']) + 
                 [0] * int(row['true_negative']) +
                 [1] * int(row['false_negative']) + 
                 [0] * int(row['false_positive']))
        
        y_score = ([1] * int(row['true_positive']) + 
                  [0] * int(row['true_negative']) +
                  [0] * int(row['false_negative']) + 
                  [1] * int(row['false_positive']))
        
        return roc_auc_score(y_true, y_score)
    except:
        return np.nan

def transform_column(source_df, target_col, source_spec):
    """
    Transform a column based on the mapping specification
    """
    if isinstance(source_spec, str):
        if source_spec.startswith('CONST='):
            return pd.Series([source_spec.split('=')[1].strip('"')] * len(source_df))
        else:
            return source_df[source_spec]
    elif isinstance(source_spec, str) and source_spec.startswith('LAMBDA='):
        # Execute lambda expression in the context of the source dataframe
        lambda_expr = source_spec.split('=')[1]
        return eval(f"source_df.apply(lambda row: {lambda_expr}, axis=1)")
    return pd.Series([np.nan] * len(source_df))

def main():
    # Read input files
    INPUT_STANDARDLLM_FILEPATH = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                             'aggregation_evaluation_results_long_final_seed7_20250128.csv')
    INPUT_AGENTICSIMLLM_FILEPATH = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                               'transcripts_stat_summary_final_20250128_013836.csv')
    
    standardllm_df = pd.read_csv(INPUT_STANDARDLLM_FILEPATH)
    agenticsimllm_df = pd.read_csv(INPUT_AGENTICSIMLLM_FILEPATH)
    
    # Column mapping dictionary
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
    
    # Process each input dataset
    dfs = []
    
    # Process INPUT_A (standardllm_df)
    input_a_columns = {}
    for col, mapping in column_map_dt.items():
        if 'INPUT_A' in mapping:
            input_a_columns[col] = transform_column(standardllm_df, col, mapping['INPUT_A'])
    dfs.append(pd.DataFrame(input_a_columns))
    
    # Process INPUT_B (agenticsimllm_df)
    input_b_columns = {}
    for col, mapping in column_map_dt.items():
        if 'INPUT_B' in mapping:
            input_b_columns[col] = transform_column(agenticsimllm_df, col, mapping['INPUT_B'])
    dfs.append(pd.DataFrame(input_b_columns))
    
    # Combine datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined dataset
    output_filepath = os.path.join('..', 'aggregation_evaluations-transcripts_manual',
                                 'combined_evaluation_results_final_seed7_20250128.csv')
    combined_df.to_csv(output_filepath, index=False)
    
    print(f"Combined data saved to: {output_filepath}")
    print(f"Total rows: {len(combined_df)}")
    print("\nSample of combined data:")
    print(combined_df.head())

if __name__ == "__main__":
    main()