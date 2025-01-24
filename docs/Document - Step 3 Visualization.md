=====[ 20250123-0140 ]=====

###CODE:
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

###INPUT_CSV_NEW:
model_name,prompt_type,total_attempts,successful_attempts,failed_attempts,timeout_attempts,execution_time_mean,execution_time_median,execution_time_sd,prediction_accuracy,auc_roc,txt_accuracy,txt_auc_roc,total_duration_mean,total_duration_median,total_duration_sd,load_duration_mean,load_duration_median,load_duration_sd,prompt_eval_duration_mean,prompt_eval_duration_median,prompt_eval_duration_sd,eval_duration_mean,eval_duration_median,eval_duration_sd,prompt_eval_count_mean,prompt_eval_count_median,prompt_eval_count_sd,eval_count_mean,eval_count_median,eval_count_sd,true_positives,true_negatives,false_positives,false_negatives,precision,recall,f1_score,model_dir,model_params,model_quantization,total_duration_sec_missing_count,load_duration_sec_missing_count,prompt_eval_duration_sec_missing_count,eval_duration_sec_missing_count,python_api_duration_sec_missing_count,confidence_txt_missing_count,prompt_eval_count_missing_count,eval_count_missing_count
athene-v2:72b-q4_K_M,cot,100,100,0,0,62.12971167564392,62.02094542980194,6.611031971496379,60.0,0.5040476190476191,60.0,0.5040476190476191,62127847044.42,,6611018249.627314,21222812.93,,5780894.085105869,634590000.0,,36973672.04163557,61464570000.0,,6599425896.755541,330.6,,8.430344421831593,425.93,,45.46355072411794,24,36,34,6,0.41379310344827586,0.8,0.5454545454545454,athene-v2_72b-q4_K_M,72b,q4_K_M,0,0,0,0,0,0,0,0
athene-v2:72b-q4_K_M,cot_nshot,72,72,0,0,58.529864854282806,58.532861828804016,7.509692119399419,63.888888888888886,0.4657118055555556,63.888888888888886,0.4657118055555556,58527805427.27778,,7509858180.211873,23019353.736111112,,6584188.083747469,4390652777.777778,,11968455.866871284,54082708333.333336,,7509024291.819866,2048.0,,0.0,362.7638888888889,,50.30988284049268,16,30,18,8,0.47058823529411764,0.6666666666666666,0.5517241379310345,athene-v2_72b-q4_K_M,72b,q4_K_M,0,0,0,0,0,0,0,0
athene-v2:72b-q4_K_M,system1,100,100,0,0,3.4656481552124023,2.99935519695282,4.634826238400708,65.0,0.6496881496881496,65.0,0.6496881496881496,3463868814.58,,4634803742.407762,476478605.0,,4550750283.802975,496280000.0,,38582478.88313653,2480600000.0,,57125946.271144934,251.95,,8.681077762070279,18.0,,0.0,20,45,29,6,0.40816326530612246,0.7692307692307693,0.5333333333333333,athene-v2_72b-q4_K_M,72b,q4_K_M,0,0,0,0,0,0,0,0
aya-expanse:8b-q4_K_M,system1,100,100,0,0,0.33005881547927857,0.25283265113830566,0.7700538172748062,52.0,0.5859640602234094,52.0,0.5859640602234094,328153139.85,,769873026.9268427,110338163.73,,690771980.8918929,12200000.0,,14862331.206405303,195150000.0,,8425999.925674979,257.51,,9.495873638671492,18.0,,0.0,19,33,38,10,0.3333333333333333,0.6551724137931034,0.4418604651162791,aya-expanse_8b-q4_K_M,8b,q4_K_M,0,0,0,0,0,0,0,0
aya-expanse:8b-q4_K_M,cot,100,100,0,0,2.320480008125305,2.268361806869507,0.31304800182283676,37.0,0.5133239831697054,37.0,0.5133239831697054,2318354462.87,,313057621.55502486,41634726.94,,5810989.030656605,15260000.0,,1194431.524477928,2257200000.0,,313150914.0167005,332.91,,8.465133480842747,233.9,,32.79643365681794,28,9,60,3,0.3181818181818182,0.9032258064516129,0.4705882352941176,aya-expanse_8b-q4_K_M,8b,q4_K_M,0,0,0,0,0,0,0,0
aya-expanse:8b-q4_K_M,cot_nshot,100,100,0,0,3.045493018627167,3.046053886413574,0.4776212917526271,35.0,0.5043859649122808,35.0,0.5043859649122808,3043096699.89,,477512040.70625204,40984782.5,,5399699.773824328,413750000.0,,8069615.038444329,2558040000.0,,476291607.47753614,2048.0,,0.0,250.2,,48.42718161520366,19,16,60,5,0.24050632911392406,0.7916666666666666,0.36893203883495146,aya-expanse_8b-q4_K_M,8b,q4_K_M,0,0,0,0,0,0,0,0
...

###MODEL_LS:
MODEL_SUBSET_LS = ['athene-v2:72b-q4_K_M','aya-expanse:8b-q4_K_M']
MODEL_ALL_LS = ['athene-v2:72b-q4_K_M','aya-expanse:8b-q4_K_M','command-r:35b-08-2024-q4_K_M','dolphin3:8b-llama3.1-q4_K_M']

###INSTRUCTIONS:

Please carefully analyze ###CODE and modify it to process the new ###INPUT_CSV_NEW to accommodate the following changes:
1. there are only 2 ensembles, MODEL_SUBSET_LS and MODEL_ALL_LS 
2. each model_name in MODEL_SUBSET_LS or MODEL_ALL_LS has 3 rows for 3 types of prompts and should be processed and plotted as 3 different models (e.g. f"{model_name}_({prompt_type}) where prompt_type in ['system1','cot','cot-nshot']
3. INPUT_FILEPATH = os.path.join('evaluation_reports_summary','summary_all_all-standardllms.csv')
4. OUTPUT_ROOT_DIR = os.path.join('evaluation_reports_summary','plots') where the 4 types of output files go (3 *.png bar plots and 1 *.md table)

===========================