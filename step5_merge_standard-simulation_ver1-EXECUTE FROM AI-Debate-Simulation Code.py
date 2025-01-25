import os
import pandas as pd

# Global variables
INPUT_ROOT_DIR = os.path.join('..', 'summary_reports')
INPUT_STANDARDLLM_FILENAME = 'transcripts_standardllm.csv'
INPUT_AGENTICSIMLLM_FILENAME = 'transcripts_agenticsimllm.csv'
OUTPUT_MERGED_FILENAME = 'transcripts_merged.csv'

# Read input files
standardllm_df = pd.read_csv(os.path.join(INPUT_ROOT_DIR, INPUT_STANDARDLLM_FILENAME))
agenticsimllm_df = pd.read_csv(os.path.join(INPUT_ROOT_DIR, INPUT_AGENTICSIMLLM_FILENAME))

# Merge model_name and prompt_type in standardllm_df
standardllm_df['model_name'] = standardllm_df.apply(
    lambda row: f"{row['model_name']}-{row['prompt_type'].split('.')[1].lower()}", 
    axis=1
)
standardllm_df = standardllm_df.drop('prompt_type', axis=1)

# Select columns from standardllm_df
standard_columns = [
    'model_name', 'model_params', 'model_quantization',
    'successful_attempts', 'failed_attempts', 'execution_time_median',
    'prediction_accuracy', 'total_duration_median', 'prompt_eval_duration_median',
    'eval_duration_median', 'true_positives', 'true_negatives',
    'false_positives', 'false_negatives', 'f1_score'
]

# Select columns from agenticsimllm_df
agenticsim_columns = [
    'model_name', 'model_params', 'model_quantization',
    'successful_attempts', 'failed_attempts', 'execution_time_median',
    'prediction_accuracy', 'total_duration_median', 'prompt_eval_duration_median',
    'eval_duration_median', 'true_positives', 'true_negatives',
    'false_positives', 'false_negatives', 'f1_score',
    'prompt_eval_count_median', 'eval_count_median', 'confidence_txt_missing_count'
]

# Create merged dataframe
standard_selected = standardllm_df[standard_columns].copy()
agenticsim_selected = agenticsimllm_df[agenticsim_columns].copy()

# Rename columns with prefixes
standard_selected.columns = ['standard_' + col for col in standard_selected.columns]
agenticsim_selected.columns = ['agenticsim_' + col for col in agenticsim_selected.columns]

# Merge dataframes
merged_df = pd.concat([standard_selected, agenticsim_selected], axis=1)

# Save merged dataframe
merged_df.to_csv(os.path.join(INPUT_ROOT_DIR, OUTPUT_MERGED_FILENAME), index=False)