import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc_roc(row):
    """
    Calculate AUC-ROC from confusion matrix values in a row
    
    Parameters:
        row: pandas Series containing confusion matrix values
        
    Returns:
        float: AUC-ROC score or np.nan if calculation fails
    """
    try:
        # Create binary arrays for true labels and predicted scores
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

def execute_lambda_function(source_df, lambda_expr):
    """
    Execute a lambda expression on the source dataframe
    
    Parameters:
        source_df: pandas DataFrame containing source data
        lambda_expr: string containing the lambda expression to execute
        
    Returns:
        pandas Series: Results of lambda expression execution
    """
    try:
        # Replace any sum() functions with explicit pandas operations
        if 'sum(' in lambda_expr:
            return eval(f"source_df.apply(lambda row: {lambda_expr}, axis=1)")
        # Handle special case for AUC-ROC calculation
        elif 'calculate_auc_roc' in lambda_expr:
            return source_df.apply(calculate_auc_roc, axis=1)
        # Handle basic mathematical operations
        else:
            return eval(f"source_df.apply(lambda row: {lambda_expr}, axis=1)")
    except Exception as e:
        print(f"Error executing lambda expression: {lambda_expr}")
        print(f"Error details: {str(e)}")
        return pd.Series([np.nan] * len(source_df))

def main():
    # Step 1: Read input files
    INPUT_STANDARDLLM_FILEPATH = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                             'aggregation_summary_report.csv')
    INPUT_AGENTICSIMLLM_FILEPATH = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                               'transcripts_stat_summary_final_20250128_013836.csv')

    OUTPUT_FILEPATH = os.path.join('..', 'aggregation_evaluations-transcripts_manual-copy',
                                 'combined_evaluations-transcripts_manual.csv')

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
    
    # Step 2: Create empty DataFrame with desired columns
    combined_df = pd.DataFrame(columns=column_map_dt.keys())
    
    # Step 3 & 4: Process INPUT_A (standardllm_df)
    input_a_data = []
    for _, row in standardllm_df.iterrows():
        new_row = {}
        for col_name, mapping in column_map_dt.items():
            if 'INPUT_A' in mapping:
                value = mapping['INPUT_A']
                if value.startswith('LAMBDA='):
                    # Handle lambda expression
                    lambda_expr = value.split('=', 1)[1]
                    new_row[col_name] = execute_lambda_function(pd.DataFrame([row]), lambda_expr).iloc[0]
                elif value.startswith('CONST='):
                    # Handle constant value
                    const_value = value.split('=', 1)[1].strip('"')
                    new_row[col_name] = const_value
                else:
                    # Handle direct column mapping
                    new_row[col_name] = row[value]
        input_a_data.append(new_row)
    
    # Step 5: Process INPUT_B (agenticsimllm_df)
    input_b_data = []
    for _, row in agenticsimllm_df.iterrows():
        new_row = {}
        for col_name, mapping in column_map_dt.items():
            if 'INPUT_B' in mapping:
                value = mapping['INPUT_B']
                if value.startswith('LAMBDA='):
                    # Handle lambda expression
                    lambda_expr = value.split('=', 1)[1]
                    new_row[col_name] = execute_lambda_function(pd.DataFrame([row]), lambda_expr).iloc[0]
                elif value.startswith('CONST='):
                    # Handle constant value
                    const_value = value.split('=', 1)[1].strip('"')
                    new_row[col_name] = const_value
                else:
                    # Handle direct column mapping
                    new_row[col_name] = row[value]
        input_b_data.append(new_row)
    
    # Combine processed data
    combined_df = pd.concat([
        pd.DataFrame(input_a_data),
        pd.DataFrame(input_b_data)
    ], ignore_index=True)
    
    # Sort the DataFrame first by 'model_name' and then by 'prompt_type'
    combined_df = combined_df.sort_values(by=['model_name', 'prompt_type'])

    # Save combined dataset
    combined_df.to_csv(OUTPUT_FILEPATH, index=False)
    
    print(f"Combined data saved to: {OUTPUT_FILEPATH}")
    print(f"Total rows: {len(combined_df)}")
    print("\nSample of combined data:")
    print(combined_df.head())
    
    # Print column completeness statistics
    print("\nColumn completeness:")
    for col in combined_df.columns:
        non_null_count = combined_df[col].count()
        completeness = (non_null_count / len(combined_df)) * 100
        print(f"{col}: {completeness:.1f}% complete ({non_null_count}/{len(combined_df)} rows)")

if __name__ == "__main__":
    main()