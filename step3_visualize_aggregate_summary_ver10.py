#!/usr/bin/env python3
"""
step4_visualize_statistics_ver4.py

Modified version that includes:
- Model filtering based on ENSEMBLE_MODEL_LS
- Error handling for filtering
- All previous visualization improvements
"""

import logging
import sys
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configure logging
def setup_logging():
    """Configure logging to both file and console with timestamps"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

EXCLUDE_MODEL_LS = [] 
EXCLUDED_OUTPUT_LS = []
# EXCLUDE_MODEL_LS = ['llama3.3:70b-instruct-q4_K_M']
# EXCLUDED_OUTPUT_LS = ['compute'] # select from ['performance','timing','compute','table']

# Updated model list based on actual directory structure
ENSEMBLE_SIZE_LS = [
    "llama3.2:1b-instruct-q4_K_M",  # 1400
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:70b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",   
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:32b-instruct-q4_K_M", 
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
]

EMSEMBLE_SIZE_DIR = [
    "llama3_1_8b_instruct_q4_k_m",
    "llama3_2_1b_instruct_q4_k_m",
    "llama3_2_3b_instruct_q4_k_m",
    "llama3_3_70b_instruct_q4_k_m",
    "qwen2_5_0_5b_instruct_q4_k_m",
    "qwen2_5_14b_instruct_q4_k_m",
    "qwen2_5_1_5b_instruct_q4_k_m",
    "qwen2_5_32b_instruct_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
]

ENSEMBLE_OSS_LS = [
    "command-r:35b-08-2024-q4_K_M",
    "gemma2:9b-instruct-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "granite3.1-moe:3b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "phi4:14b-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",      # in previous SIZE ensemble
    "llama3.1:8b-instruct-q4_K_M",     # in previous SIZE ensemble
]
ENSEMBLE_OSS_DIR = [
    "command_r_35b_08_2024_q4_k_m",
    "gemma2_9b_instruct_q4_k_m",
    "granite3_1_dense_8b_instruct_q4_k_m",
    "granite3_1_moe_3b_instruct_q4_k_m",
    "mistral_7b_instruct_q4_k_m",
    "phi4_14b_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m",
    "llama3_1_8b_instruct_q4_k_m",
]

ENSEMBLE_REASONING_LS = [
    "athene-v2:72b-q4_K_M",
    "dolphin3:8b-llama3.1-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "falcon3:7b-instruct-q4_K_M",        # 36
    "glm4:9b-chat-q4_K_M",
    "hermes3:8b-llama3.1-q4_K_M",        # 146 (runpod 03:07)
    "marco-o1:7b-q4_K_M",
    "nemotron-mini:4b-instruct-q4_K_M",  # 0
    "olmo2:7b-1124-instruct-q4_K_M",
    "smallthinker:3b-preview-q4_K_M",    # 104
    "tulu3:8b-q4_K_M",
]
ENSEMBLE_REASONING_DIR = [
    "athene_v2_72b_q4_k_m",
    "dolphin3_8b_llama3_1_q4_k_m",
    "exaone3_5_7_8b_instruct_q4_k_m",
    "falcon3_7b_instruct_q4_k_m",
    "glm4_9b_chat_q4_k_m",
    "hermes3_8b_llama3_1_q4_k_m",
    "marco_o1_7b_q4_k_m",
    "nemotron_mini_4b_instruct_q4_k_m",
    "olmo2_7b_1124_instruct_q4_k_m",
    "smallthinker_3b_preview_q4_k_m",
    "tulu3_8b_q4_k_m",
] 
# Add these from previous SIZE ensemble
#     "qwen2.5:7b-instruct-q4_K_M",
#     "llama3.1:8b-instruct-q4_K_M",
# Add these from previous REASONING ensemble
#     "command-r:35b-08-2024-q4_K_M",
#     "falcon3:7b-instruct-q4_K_M",
#     "gemma2:9b-instruct-q4_K_M",
#     "granite3.1-dense:8b-instruct-q4_K_M",
#     "llama3.1:8b-instruct-q4_K_M",
#     "marco-o1:7b-q4_K_M",
#     "phi4:14b-q4_K_M",
#     "qwen2.5:7b-instruct-q4_K_M",
#     "tulu3:8b-q4_K_M",
# Add the LARGE REASONING models
#     "tulu3:8b-q4_K_M",
#     "qwq:32b-preview-q4_K_M",
#     "qwen2.5:72b-instruct-q4_K_M",
#     "reflection:70b-q4_K_M",

ENSEMBLE_ALL_LS = [
    "athene-v2:72b-q4_K_M",
    "aya-expanse:8b-q4_K_M",
    "command-r:35b-08-2024-q4_K_M",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "dolphin3:8b-llama3.1-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "falcon3:7b-instruct-q4_K_M",        # 36
    "gemma2:9b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "granite3.1-moe:3b-instruct-q4_K_M",
    "hermes3:8b-llama3.1-q4_K_M",        # 146 (runpod 03:07)
    "llama3.1:70b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-fp16",
    "llama3.2:1b-instruct-q4_K_M",  # 1400
    "llama3.2:1b-instruct-fp16", 
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-fp16",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",  
    "marco-o1:7b-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "nemotron-mini:4b-instruct-q4_K_M",  # 0
    "olmo2:7b-1124-instruct-q4_K_M",
    "phi4:14b-q4_K_M",
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:32b-instruct-q4_K_M", 
    "qwen2.5:72b-instruct-q4_K_M", 
    "qwq:32b-preview-q4_K_M",
    "smallthinker:3b-preview-q4_K_M",    # 104
    "tulu3:8b-q4_K_M",
]

ENSEMBLE_ALL_DIR = [
    "athene_v2_72b_q4_k_m",
    "aya_expanse_8b_q4_k_m",
    "command_r_35b_08_2024_q4_K_M",
    "deepseek_r1_1_5b",
    "deepseek_r1_7b",
    "deepseek_r1_8b",
    "deepseek_r1_14b",
    "deepseek_r1_32b",
    "deepseek_r1_70b",
    "dolphin3_8b_llama3_1_q4_k_m",
    "exaone3_5_7_8b_instruct_q4_k_m",
    "falcon3_7b_instruct_q4_k_m",
    "gemma2_9b_instruct_q4_k_m",
    "glm4_9b_chat_q4_k_m",
    "granite3_1_dense_8b_instruct_q4_k_m",
    "granite3_1_moe_3b_instruct_q4_k_m",
    "hermes3_8b_llama3_1_q4_k_m",
    "llama3_1_70b_instruct_q4_k_m",
    "llama3_1_8b_instruct_fp16",
    "llama3_1_8b_instruct_q4_k_m",
    "llama3_2_1b_instruct_fp16",
    "llama3_2_1b_instruct_q4_k_m",
    "llama3_2_3b_instruct_fp16",
    "llama3_2_3b_instruct_q4_k_m",
    "llama3_3_70b_instruct_q4_k_m",
    "marco_o1_7b_q4_k_m",
    "mistral_7b_instruct_q4_k_m",
    "nemotron_mini_4b_instruct_q4_k_m",
    "olmo2_7b_1124_instruct_q4_k_m",
    "phi4_14b_q4_k_m",
    "qwen2_5_0_5b_instruct_q4_k_m",
    "qwen2_5_14b_instruct_q4_k_m",
    "qwen2_5_1_5b_instruct_q4_k_m",
    "qwen2_5_32b_instruct_q4_k_m",
    "qwen2_5_3b_instruct_q4_k_m_partial-34",
    "qwen2_5_72b_instruct_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m",
    "qwq_32b_preview_q4_partial-140",
    "smallthinker_3b_preview_q4_k_m",
    "tulu3_8b_q4_k_m",
]

ENSEMBLE_NAME = 'all'
DATETIME_STR = '20250123_151830'
ENSEMBLE_TYPE_DT = {'size': ENSEMBLE_SIZE_LS, 
                    'oss': ENSEMBLE_OSS_LS,
                    'reasoning': ENSEMBLE_REASONING_LS,
                    'all': ENSEMBLE_ALL_LS}
ENSEMBLE_DESCRIPTION_DT = {'size': "Ensemble by Size of LLM",
                    'oss': "Ensemble by Leading Open LLMs",
                    'reasoning': "Ensemble by Specialized Reasoning LLMs",
                    'all': "All LLM Models"}

ENSEMBLE_TYPE_NAME = ENSEMBLE_TYPE_DT[ENSEMBLE_NAME]
ENSEMBLE_MODEL_LS = ENSEMBLE_TYPE_NAME

def filter_and_validate_models(df, model_list):
    """
    Filters DataFrame to include only specified models that exist in the data.
    Includes detailed logging of the filtering process.
    """
    logger.info(f"Starting model filtering process")
    logger.info(f"Total number of models in original data: {len(df['model'].unique())}")
    
    if not model_list:
        logger.info("No model list provided - using all available models")
        return df
    
    logger.info(f"Number of models in requested model list: {len(model_list)}")
    
    # Find which models from our list actually exist in the data
    available_models = set(df['model'])
    valid_models = [model for model in model_list if model in available_models]
    
    # Detailed logging of model availability
    logger.info("\nChecking model availability:")
    for model in model_list:
        if model in available_models:
            logger.info(f"✓ Found model: {model}")
        else:
            logger.warning(f"✗ Missing model: {model}")
    
    if not valid_models:
        error_msg = "None of the specified models found in the data"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Print information about missing models
    missing_models = set(model_list) - available_models
    if missing_models:
        logger.warning(f"\nThe following {len(missing_models)} models were not found in the data:")
        for model in sorted(missing_models):
            logger.warning(f"  - {model}")
    
    # Filter DataFrame
    df_filtered = df[df['model'].isin(valid_models)].reset_index(drop=True)
    logger.info(f"\nFiltering complete:")
    logger.info(f"  - Original model count: {len(df)}")
    logger.info(f"  - Filtered model count: {len(df_filtered)}")
    
    return df_filtered

def create_performance_plot(df, output_path, root_filename):
    """Creates a bar plot for model performance metrics with detailed logging."""
    logger.info("\nCreating performance plot...")
    
    try:
        # Filter models and log progress
        logger.info("Filtering and sorting models by f1_score")
        df_filtered = filter_and_validate_models(df, ENSEMBLE_MODEL_LS)

        # Remove any models from the excluded list if 'performance' in targeted output files
        if "performance" in EXCLUDED_OUTPUT_LS:
            df_filtered = df_filtered[~df_filtered['model'].isin(EXCLUDE_MODEL_LS)]

        df_sorted = df_filtered.sort_values('f1_score', ascending=False).reset_index(drop=True)
        
        logger.info("Creating normalized accuracy calculations")
        plot_data = pd.DataFrame({
            'model': df_sorted['model'],
            'f1_score': df_sorted['f1_score'],
            'accuracy_normalized': df_sorted['accuracy'] / 100
        })
        
        logger.info("Preparing data for plotting")
        perf_melt = plot_data.melt(
            id_vars="model",
            value_vars=["f1_score", "accuracy_normalized"],
            var_name="metric",
            value_name="value"
        )
        
        logger.info("Generating plot")
        plt.figure(figsize=(10, 6))
        sns.barplot(x="model", y="value", hue="metric", data=perf_melt)
        
        # Customize plot
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.title(f"Performance Metrics by Model ({ENSEMBLE_DESCRIPTION_DT[ENSEMBLE_NAME]})")
        plt.ylim(0, 1)
        plt.legend(title="Metric", labels=["F1 Score", "Accuracy"])
        plt.tight_layout()
        
        # Save plot
        output_filename = f"{root_filename}_performance.png"
        output_fullpath = os.path.join(output_path, output_filename)
        plt.savefig(output_fullpath)
        plt.close()
        logger.info(f"Performance plot saved successfully to {output_fullpath}")
        
    except Exception as e:
        logger.error(f"Error creating performance plot: {str(e)}")
        plt.close()
        raise

def create_timing_plot(df, output_path, root_filename):
    """
    Creates a stacked bar plot for timing metrics.
    Only includes models specified in ENSEMBLE_MODEL_LS.
    """
    try:
        # Filter models and calculate total time
        df_filtered = filter_and_validate_models(df, ENSEMBLE_MODEL_LS)
        df_filtered['total_time'] = df_filtered['execution_time_mean'] + df_filtered['eval_duration_mean']

        # Remove any models from the excluded list if 'timing' in targeted output files
        if "timing" in EXCLUDED_OUTPUT_LS:
            df_filtered = df_filtered[~df_filtered['model'].isin(EXCLUDE_MODEL_LS)]
            logger.info(f"Excluded models removed from timing plot: {EXCLUDE_MODEL_LS}")

        df_sorted = df_filtered.sort_values('f1_score', ascending=False).reset_index(drop=True)


        df_sorted = df_filtered.sort_values('total_time', ascending=False).reset_index(drop=True)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create stacked bars
        plt.bar(df_sorted['model'], df_sorted['eval_duration_mean'],
                label='Evaluation Duration')
        plt.bar(df_sorted['model'], df_sorted['execution_time_mean'],
                bottom=df_sorted['eval_duration_mean'],
                label='Execution Time')
        
        # Customize the plot
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.title(f"Compute (Timing) Metrics by Model ({ENSEMBLE_DESCRIPTION_DT[ENSEMBLE_NAME]})")
        plt.xlabel("Model")
        plt.ylabel("Time")
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        output_filename = f"{root_filename}_compute-timing.png"
        output_fullpath = os.path.join(output_path, output_filename)
        plt.savefig(output_fullpath)
        plt.close()
        print(f"Timing plot saved to {output_fullpath}")
        
    except Exception as e:
        print(f"Error creating timing plot: {str(e)}")
        plt.close()
        raise

def create_token_plot(df, output_path, root_filename):
    """
    Creates a stacked bar plot for compute metrics.
    Only includes models specified in ENSEMBLE_MODEL_LS.
    """
    try:
        # Filter models and calculate total count
        df_filtered = filter_and_validate_models(df, ENSEMBLE_MODEL_LS)
        df_filtered['total_count'] = df_filtered['prompt_eval_count_mean'] + df_filtered['eval_count_mean']

        # Remove any models from the excluded list if 'compute' in targeted output files
        if "compute" in EXCLUDED_OUTPUT_LS:
            df_filtered = df_filtered[~df_filtered['model'].isin(EXCLUDE_MODEL_LS)]

        df_sorted = df_filtered.sort_values('f1_score', ascending=False).reset_index(drop=True)


        df_sorted = df_filtered.sort_values('total_count', ascending=False).reset_index(drop=True)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create stacked bars
        plt.bar(df_sorted['model'], df_sorted['prompt_eval_count_mean'],
                label='Prompt Evaluation Count')
        plt.bar(df_sorted['model'], df_sorted['eval_count_mean'],
                bottom=df_sorted['prompt_eval_count_mean'],
                label='Evaluation Count')
        
        # Customize the plot
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.title(f"Compute (Token) Metrics by Model ({ENSEMBLE_DESCRIPTION_DT[ENSEMBLE_NAME]})")
        plt.xlabel("Model")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        output_filename = f"{root_filename}_compute-tokens.png"
        output_fullpath = os.path.join(output_path, output_filename)
        plt.savefig(output_fullpath)
        plt.close()
        print(f"Compute plot saved to {output_fullpath}")
        
    except Exception as e:
        print(f"Error creating compute plot: {str(e)}")
        plt.close()
        raise

def generate_markdown_table(df, output_path, root_filename):
    """
    Generates a markdown table with reordered columns and sorted by f1_score.
    Only includes models specified in ENSEMBLE_MODEL_LS.
    """
    try:
        # Filter models and sort by f1_score
        df_filtered = filter_and_validate_models(df, ENSEMBLE_MODEL_LS)

        # Remove any models from the excluded list if 'table' in targeted output files
        if "table" in EXCLUDED_OUTPUT_LS:
            df_filtered = df_filtered[~df_filtered['model'].isin(EXCLUDE_MODEL_LS)]

        df_sorted = df_filtered.sort_values('f1_score', ascending=False).reset_index(drop=True)

        df_sorted = df_filtered.sort_values('f1_score', ascending=False).reset_index(drop=True)
        
        # Generate markdown lines
        md_lines = []
        md_lines.append("# Model Performance and Compute Summary")
        md_lines.append("")
        
        # Column headers
        md_lines.append("| model | f1_score | accuracy | precision | recall | execution_time_mean | execution_time_sd | eval_duration_mean | prompt_eval_count_mean | eval_count_mean |")
        md_lines.append("|-------|-----------|----------|-----------|--------|-------------------|------------------|-------------------|----------------------|-----------------|")

        # Generate table rows
        for _, row in df_sorted.iterrows():
            md_lines.append(
                f"| {row['model']} "
                f"| {row['f1_score']:.4f} "
                f"| {row['accuracy']:.4f} "
                f"| {row['precision']:.4f} "
                f"| {row['recall']:.4f} "
                f"| {row['execution_time_mean']:.4f} "
                f"| {row['execution_time_sd']:.4f} "
                f"| {row['eval_duration_mean']:.4f} "
                f"| {row['prompt_eval_count_mean']:.2f} "
                f"| {row['eval_count_mean']:.2f} |"
            )

        # Write to file
        output_filename = f"{root_filename}_table.md"
        output_fullpath = os.path.join(output_path, output_filename)
        with open(output_fullpath, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        print(f"Markdown summary written to {output_fullpath}")
        
    except Exception as e:
        print(f"Error generating markdown table: {str(e)}")
        raise



def validate_paths(input_path, output_path):
    """
    Validates input and output paths exist and are accessible.
    
    Args:
        input_path (str): Full path to input JSON file
        output_path (str): Directory path for output files
    
    Returns:
        tuple: (input_path, output_path)
    """
    logger.info("Validating file paths...")
    
    # Check if input file exists
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        logger.info("Available files in directory:")
        parent_dir = os.path.dirname(input_path)
        if os.path.exists(parent_dir):
            files = os.listdir(parent_dir)
            for file in sorted(files):
                if file.endswith('.json'):
                    logger.info(f"  - {file}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Check if output directory exists, create if it doesn't
    if not os.path.exists(output_path):
        logger.warning(f"Output directory does not exist, creating: {output_path}")
        os.makedirs(output_path)
    
    logger.info("Path validation successful")
    logger.info(f"  - Input file: {input_path}")
    logger.info(f"  - Output directory: {output_path}")
    
    return input_path, output_path


def main():
    """Main execution function that orchestrates the entire visualization process"""
    logger.info("Starting visualization process")
    
    try:
        # Part 1: Configuration Setup
        # These values determine which model ensemble to analyze and what timestamp to use
        # ENSEMBLE_NAME = 'all'  # Specifies we want to analyze all models
        # DATETIME_STR = '20250123_133911'  # Fixed timestamp matching our input data
        logger.info(f"Configuration:")
        logger.info(f"  - Ensemble: {ENSEMBLE_NAME}")
        logger.info(f"  - DateTime: {DATETIME_STR}")
        
        # Part 2: Path Configuration and Validation
        # Define input and output paths using the configuration values
        INPUT_JSON_FILENAME = f"transcripts_stat_summary_all_{DATETIME_STR}.json"
        INPUT_JSON_FULLPATH = os.path.abspath(os.path.join("..", "summary_reports", INPUT_JSON_FILENAME))
        OUTPUT_FULLPATH = os.path.abspath(os.path.join("..", "summary_reports"))
        
        # Validate all paths exist and are accessible before proceeding
        INPUT_JSON_FULLPATH, OUTPUT_FULLPATH = validate_paths(INPUT_JSON_FULLPATH, OUTPUT_FULLPATH)

        # Part 3: Data Loading
        # Read and parse the JSON data with careful error handling
        logger.info("Reading input JSON file...")
        try:
            with open(INPUT_JSON_FULLPATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info("JSON data loaded successfully")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading file: {str(e)}")
            raise

        # Part 4: Data Extraction and Processing
        # Extract performance and compute metrics for each model
        logger.info("Extracting performance and compute data")
        performance_dict = data["performance"]["models"]

        # Check if any excluded models don't exist in performance_dict
        for excluded_model in EXCLUDE_MODEL_LS:
            if excluded_model not in performance_dict:
                logger.warning(f"Note: Excluded model '{excluded_model}' not found in performance dictionary")

        
        compute_dict = data["compute"]["models"]
        logger.info(f"Found {len(performance_dict)} models in performance data")
        logger.info(f"Found {len(compute_dict)} models in compute data")

        # Part 5: DataFrame Creation
        # Create a consolidated DataFrame combining all metrics
        logger.info("Creating consolidated DataFrame")
        rows = []

        # Modified loop with exclusion logic
        for model_name in performance_dict:
            # Skip if model is in exclusion list
            if model_name in EXCLUDE_MODEL_LS:
                logger.info(f"Skipping excluded model: {model_name}")
                continue
            
            # Original loop content continues here
            logger.info(f"Processing model: {model_name}")
            perf = performance_dict.get(model_name, {})
            comp = compute_dict.get(model_name, {})
            
            # Create a row with all metrics for this model
            row = {
                "model": model_name,
                "accuracy": perf.get("accuracy", None),
                "precision": perf.get("precision", None),
                "recall": perf.get("recall", None),
                "f1_score": perf.get("f1_score", None),
                "execution_time_mean": comp.get("execution_time_mean", None),
                "execution_time_sd": comp.get("execution_time_sd", None),
                "eval_duration_mean": comp.get("eval_duration_mean", None),
                "prompt_eval_count_mean": comp.get("prompt_eval_count_mean", None),
                "eval_count_mean": comp.get("eval_count_mean", None)
            }
            rows.append(row)
        
        # Convert list of rows to DataFrame
        df = pd.DataFrame(rows)
        logger.info(f"DataFrame created successfully with {len(df)} rows")

        # Part 6: Visualization Setup
        # Configure the visual style for all plots
        logger.info("Setting up visualization environment")
        sns.set_theme(style="whitegrid")

        # Part 7: Output Generation
        # Create root filename for all outputs using configuration values
        ROOT_FILENAME = f"transcripts_stat_visualization_{ENSEMBLE_NAME}_{DATETIME_STR}"
        logger.info("Generating all visualization outputs")
        logger.info(f"Root filename for outputs: {ROOT_FILENAME}")

        # Generate each type of output
        logger.info("Creating markdown table...")
        generate_markdown_table(df, OUTPUT_FULLPATH, ROOT_FILENAME)
        
        logger.info("Creating performance plot...")
        create_performance_plot(df, OUTPUT_FULLPATH, ROOT_FILENAME)
        
        logger.info("Creating timing plot...")
        create_timing_plot(df, OUTPUT_FULLPATH, ROOT_FILENAME)
        
        logger.info("Creating compute plot...")
        create_token_plot(df, OUTPUT_FULLPATH, ROOT_FILENAME)

        logger.info("All visualization tasks completed successfully!")
        
    except Exception as e:
        # Catch and log any unexpected errors that occur during execution
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)