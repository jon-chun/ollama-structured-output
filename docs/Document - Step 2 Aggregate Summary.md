# Model Evaluation Results Aggregator
## Technical Overview

This system aggregates and analyzes evaluation results from multiple ML model runs, combining metrics from paired JSON and TXT report files into a consolidated CSV output.

### Core Architecture

The codebase is organized into six main sections:
1. Global Constants
2. Logging Setup
3. Utility Functions
4. Parsing Functions
5. Core Aggregation
6. Main Execution

### Key Components

#### Data Flow
```
[Model Directories]
    └── reports/
        ├── metrics_*.json  # Core metrics and metadata
        └── report_*.txt    # Summary statistics
        
    ↓ (Processing)
    
[Consolidated Output]
    └── aggregate_reports/
        └── aggregate_model_reports.csv
```

#### File Format Requirements

1. JSON Reports (`metrics_*.json`) contain:
   - Model identification
   - Attempt statistics
   - Execution timing
   - Confusion matrix
   - Meta-data averages

2. TXT Reports (`report_*.txt`) contain:
   - Accuracy percentages
   - AUC-ROC scores

### Key Features

1. **Flexible Model Naming**: Extracts model parameters and quantization info from directory names using regex
   - Example: `llama3_2_1b-instruct-fp16` → {params: "1", quant: "fp16"}

2. **Comprehensive Metrics**:
   - Execution statistics (success, failure, timeout rates)
   - Performance metrics (accuracy, AUC-ROC, precision, recall, F1)
   - Timing data (load duration, eval duration, etc.)

3. **Robust Error Handling**:
   - Graceful handling of missing files
   - Detailed logging at multiple levels
   - Validation of parsed values

### Usage

1. Set the input directory in `EVALUATION_RESULTS_DIR`
2. Run the script
3. Find outputs in `aggregate_reports/{input_dir_name}/`

### Expected Directory Structure

```
evaluation_results_long_h100_20250115/
├── model1_directory/
│   └── reports/
│       ├── metrics_*.json
│       └── report_*.txt
├── model2_directory/
│   └── reports/
...
```

### Key Functions

- `aggregate_reports()`: Main orchestrator
- `process_report_files()`: Combines JSON and TXT data
- `extract_model_metadata()`: Parses model info from directory names
- `parse_report_json/txt()`: File-specific parsers

### Logging

- Debug logs go to dated log files
- INFO level and above appear in console
- Custom logging function supports 5 levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Output

The script generates:
1. A consolidated CSV with all metrics
2. Detailed logs of the aggregation process
3. Support for future visualization additions

### Development Notes

- Add new model naming patterns to `extract_model_metadata()`
- Extend `parse_report_txt()` for new metrics
- Add visualizations in `main()` if needed


# Prompts:

=====[ 20250123-0105]=====
###INPUT_ROOT_DIR:
(venv) jonc@jonc-MS-7C35:~/code/ollama-structured-output$ ls -altr evaluation_reports_manual/
total 376
-rw-rw-r--  1 jonc jonc  1356 Jan 23 23:44 metrics_athene-v2:72b-q4_K_M_system1_20250109_105829.json
-rw-rw-r--  1 jonc jonc  1368 Jan 23 23:44 metrics_athene-v2:72b-q4_K_M_cot_20250109_124209.json
-rw-rw-r--  1 jonc jonc  1408 Jan 23 23:44 metrics_athene-v2:72b-q4_K_M_cot-nshot_20250109_142315.json
-rw-rw-r--  1 jonc jonc  1361 Jan 23 23:44 metrics_aya-expanse:8b-q4_K_M_system1_20250108_041647.json
-rw-rw-r--  1 jonc jonc  1361 Jan 23 23:44 metrics_aya-expanse:8b-q4_K_M_cot-nshot_20250108_042557.json
-rw-rw-r--  1 jonc jonc  1368 Jan 23 23:44 metrics_aya-expanse:8b-q4_K_M_cot_20250108_042046.json
-rw-rw-r--  1 jonc jonc  1383 Jan 23 23:44 metrics_command-r:35b-08-2024-q4_K_M_system1_20250108_050140.json
-rw-rw-r--  1 jonc jonc  1371 Jan 23 23:44 metrics_command-r:35b-08-2024-q4_K_M_cot-nshot_20250108_053412.json
-rw-rw-r--  1 jonc jonc  1376 Jan 23 23:44 metrics_command-r:35b-08-2024-q4_K_M_cot_20250108_051824.json
-rw-rw-r--  1 jonc jonc  1383 Jan 23 23:44 metrics_dolphin3:8b-llama3.1-q4_K_M_system1_20250108_053444.json
-rw-rw-r--  1 jonc jonc  1369 Jan 23 23:44 metrics_dolphin3:8b-llama3.1-q4_K_M_cot-nshot_20250108_054354.json
-rw-rw-r--  1 jonc jonc  1373 Jan 23 23:44 metrics_dolphin3:8b-llama3.1-q4_K_M_cot_20250108_053935.json
...


###INPUT_SYSTEM!_FILE_CONTENTS: (e.g. metrics_athene-v2:72b-q4_K_M_system1_20250109_105829.json)
{
  "prompt_type": "PromptType.SYSTEM1",
  "model_name": "athene-v2:72b-q4_K_M",
  "start_time": "2025-01-09 10:52:40.608732",
  "end_time": "2025-01-09 10:58:29.836502",
  "total_attempts": 100,
  "successful_attempts": 100,
  "failed_attempts": 0,
  "timeout_attempts": 0,
  "avg_execution_time": 3.4656481552124023,
  "median_execution_time": 2.99935519695282,
  "sd_execution_time": 4.634826238400708,
  "timeout_stats": {
    "total_timeouts": 0,
    "avg_timeout_duration": 0.0,
    "max_timeout_duration": 0.0,
    "total_timeout_duration": 0.0
  },
  "meta_data_averages": {
    "done": 1.0,
    "total_duration": 3463868814.58,
    "load_duration": 476478605.0,
    "prompt_eval_count": 251.95,
    "prompt_eval_duration": 496280000.0,
    "eval_count": 18.0,
    "eval_duration": 2480600000.0
  },
  "meta_data_sd": {
    "done": 0.0,
    "total_duration": 4634803742.407762,
    "load_duration": 4550750283.802975,
    "prompt_eval_count": 8.681077762070279,
    "prompt_eval_duration": 38582478.88313653,
    "eval_count": 0.0,
    "eval_duration": 57125946.271144934
  },
  "prediction_accuracy": 65.0,
  "prediction_distribution": {
    "NO": 51,
    "YES": 49
  },
  "actual_distribution": {
    "NO": 74,
    "YES": 26
  },
  "confusion_matrix": {
    "tp": 20,
    "tn": 45,
    "fp": 29,
    "fn": 6
  },
  "auc_roc": 0.6496881496881496
}

###INPUT_COT_FILE_CONTENTS: (e.g. metrics_athene-v2:72b-q4_K_M_cot_20250109_124209.json)
{
  "prompt_type": "PromptType.COT",
  "model_name": "athene-v2:72b-q4_K_M",
  "start_time": "2025-01-09 10:58:33.875378",
  "end_time": "2025-01-09 12:42:09.507460",
  "total_attempts": 100,
  "successful_attempts": 100,
  "failed_attempts": 0,
  "timeout_attempts": 0,
  "avg_execution_time": 62.12971167564392,
  "median_execution_time": 62.02094542980194,
  "sd_execution_time": 6.611031971496379,
  "timeout_stats": {
    "total_timeouts": 0,
    "avg_timeout_duration": 0.0,
    "max_timeout_duration": 0.0,
    "total_timeout_duration": 0.0
  },
  "meta_data_averages": {
    "done": 1.0,
    "total_duration": 62127847044.42,
    "load_duration": 21222812.93,
    "prompt_eval_count": 330.6,
    "prompt_eval_duration": 634590000.0,
    "eval_count": 425.93,
    "eval_duration": 61464570000.0
  },
  "meta_data_sd": {
    "done": 0.0,
    "total_duration": 6611018249.627314,
    "load_duration": 5780894.085105869,
    "prompt_eval_count": 8.430344421831593,
    "prompt_eval_duration": 36973672.04163557,
    "eval_count": 45.46355072411794,
    "eval_duration": 6599425896.755541
  },
  "prediction_accuracy": 60.0,
  "prediction_distribution": {
    "YES": 58,
    "NO": 42
  },
  "actual_distribution": {
    "NO": 70,
    "YES": 30
  },
  "confusion_matrix": {
    "tp": 24,
    "tn": 36,
    "fp": 34,
    "fn": 6
  },
  "auc_roc": 0.5040476190476191
}

###INPUT_COT-NSHOT_FILE_CONTENTS: (e.g. metrics_athene-v2:72b-q4_K_M_cot-nshot_20250109_142315.json)
{
  "prompt_type": "PromptType.COT_NSHOT",
  "model_name": "athene-v2:72b-q4_K_M",
  "start_time": "2025-01-09 13:12:59.562683",
  "end_time": "2025-01-09 14:23:15.660747",
  "total_attempts": 72,
  "successful_attempts": 72,
  "failed_attempts": 0,
  "timeout_attempts": 0,
  "avg_execution_time": 58.529864854282806,
  "median_execution_time": 58.532861828804016,
  "sd_execution_time": 7.509692119399419,
  "timeout_stats": {
    "total_timeouts": 0,
    "avg_timeout_duration": 0.0,
    "max_timeout_duration": 0.0,
    "total_timeout_duration": 0.0
  },
  "meta_data_averages": {
    "done": 1.0,
    "total_duration": 58527805427.27778,
    "load_duration": 23019353.736111112,
    "prompt_eval_count": 2048.0,
    "prompt_eval_duration": 4390652777.777778,
    "eval_count": 362.7638888888889,
    "eval_duration": 54082708333.333336
  },
  "meta_data_sd": {
    "done": 0.0,
    "total_duration": 7509858180.211873,
    "load_duration": 6584188.083747469,
    "prompt_eval_count": 0.0,
    "prompt_eval_duration": 11968455.866871284,
    "eval_count": 50.30988284049268,
    "eval_duration": 7509024291.819866
  },
  "prediction_accuracy": 63.888888888888886,
  "prediction_distribution": {
    "NO": 38,
    "YES": 34
  },
  "actual_distribution": {
    "NO": 48,
    "YES": 24
  },
  "confusion_matrix": {
    "tp": 16,
    "tn": 30,
    "fp": 18,
    "fn": 8
  },
  "auc_roc": 0.4657118055555556
}

###OUTPUT_CSV:

model_name,prompt_type,total_attempts,successful_attempts,failed_attempts,timeout_attempts,execution_time_mean,execution_time_median,execution_time_sd,prediction_accuracy,auc_roc,txt_accuracy,txt_auc_roc,total_duration_mean,total_duration_median,total_duration_sd,load_duration_mean,load_duration_median,load_duration_sd,prompt_eval_duration_mean,prompt_eval_duration_median,prompt_eval_duration_sd,eval_duration_mean,eval_duration_median,eval_duration_sd,prompt_eval_count_mean,prompt_eval_count_median,prompt_eval_count_sd,eval_count_mean,eval_count_median,eval_count_sd,true_positives,true_negatives,false_positives,false_negatives,precision,recall,f1_score,model_dir,model_params,model_quantization,total_duration_sec_missing_count,load_duration_sec_missing_count,prompt_eval_duration_sec_missing_count,eval_duration_sec_missing_count,python_api_duration_sec_missing_count,confidence_txt_missing_count,prompt_eval_count_missing_count,eval_count_missing_count
athene-v2:72b-q4_K_M,system1,149,149,0,0,35.27190747442282,34.709590111,2.8611504811558013,63.758389261744966,,63.758389261744966,,35.27190747442282,34.709590111,2.8611504811558013,7.757937964825504,7.804124643,0.24495467059524512,1.1704765100671142,1.182,0.0756335032201979,25.826758389261744,25.505,2.822385117598346,788.261744966443,783.0,39.54511552390993,422.5503355704698,416.0,46.05628698486862,95,0,54,0,0.6375838926174496,1.0,0.7786885245901639,athene-v2_72b-q4_K_M,1,q4,0,0,0,0,0,0,0,0
athene-v2:72b-q4_K_M,cot,149,149,0,0,35.27190747442282,34.709590111,2.8611504811558013,63.758389261744966,,63.758389261744966,,35.27190747442282,34.709590111,2.8611504811558013,7.757937964825504,7.804124643,0.24495467059524512,1.1704765100671142,1.182,0.0756335032201979,25.826758389261744,25.505,2.822385117598346,788.261744966443,783.0,39.54511552390993,422.5503355704698,416.0,46.05628698486862,95,0,54,0,0.6375838926174496,1.0,0.7786885245901639,athene-v2_72b-q4_K_M,1,q4,0,0,0,0,0,0,0,0
athene-v2:72b-q4_K_M,cot-nshot,149,149,0,0,35.27190747442282,34.709590111,2.8611504811558013,63.758389261744966,,63.758389261744966,,35.27190747442282,34.709590111,2.8611504811558013,7.757937964825504,7.804124643,0.24495467059524512,1.1704765100671142,1.182,0.0756335032201979,25.826758389261744,25.505,2.822385117598346,788.261744966443,783.0,39.54511552390993,422.5503355704698,416.0,46.05628698486862,95,0,54,0,0.6375838926174496,1.0,0.7786885245901639,athene-v2_72b-q4_K_M,1,q4,0,0,0,0,0,0,0,0
aya-expanse:8b-q4_K_M,system1,60,60,0,0,4.0463551825,3.9533854325,0.6231177798391149,53.333333333333336,,53.333333333333336,,4.0463551825,3.9533854325,0.6231177798391149,0.0561234744,0.056586724500000005,0.007314490501996379,0.08671666666666668,0.084,0.012269697641995446,3.8914666666666666,3.8135000000000003,0.6154935037084832,764.5333333333333,763.0,34.166326303940636,360.51666666666665,359.0,41.01694220712459,32,0,28,0,0.5333333333333333,1.0,0.6956521739130436,aya-expanse_8b-q4_K_M,1,q4,0,0,0,0,0,0,0,0
aya-expanse:8b-q4_K_M,cot,60,60,0,0,4.0463551825,3.9533854325,0.6231177798391149,53.333333333333336,,53.333333333333336,,4.0463551825,3.9533854325,0.6231177798391149,0.0561234744,0.056586724500000005,0.007314490501996379,0.08671666666666668,0.084,0.012269697641995446,3.8914666666666666,3.8135000000000003,0.6154935037084832,764.5333333333333,763.0,34.166326303940636,360.51666666666665,359.0,41.01694220712459,32,0,28,0,0.5333333333333333,1.0,0.6956521739130436,aya-expanse_8b-q4_K_M,1,q4,0,0,0,0,0,0,0,0
aya-expanse:8b-q4_K_M,cot-nshot,60,60,0,0,4.0463551825,3.9533854325,0.6231177798391149,53.333333333333336,,53.333333333333336,,4.0463551825,3.9533854325,0.6231177798391149,0.0561234744,0.056586724500000005,0.007314490501996379,0.08671666666666668,0.084,0.012269697641995446,3.8914666666666666,3.8135000000000003,0.6154935037084832,764.5333333333333,763.0,34.166326303940636,360.51666666666665,359.0,41.01694220712459,32,0,28,0,0.5333333333333333,1.0,0.6956521739130436,aya-expanse_8b-q4_K_M,1,q4,0,0,0,0,0,0,0,0
...

###INSTRUCTIONS:
Please craft a clear, self-documenting, and concise python program step2_aggregate_reports_manual_ver1.py that does the following:

1. Defines the global variables:
1.a. INPUT_ROOT_DIR = os.path.join('evaluation_reports_manual')
1.b. OUTPUT_ROOT_DIR = os.path.join('evaluation_reports_summary')
1.c. MODEL_LS = ["athene-v2:72b-q4_K_M","aya-expanse:8b-q4_K_M","command-r:35b-08-2024-q4_K_M"]
1.d. ENSEMBLE_NAME = "test-20250124-0053"
1.e. PROMPT_TYPE_LS = ['_cot_','_cot-nshot_','_system1_']

2. Iterates over INPUT_ROOT_DIR and processes the set of *.json files containg the substrings in PROMPT_TYPE_LS that correspond to each model_name in MODEL_LS

3. contains specialized functions to process each of the PROMPT_TYPE_LS *.json files which for now include thes functions and input file examples:
3.a. process_cot(): ###INPUT_COT_FILE_CONTENTS
3.b. process_cot-nshot(): ###INPUT_COT-NSHOT_FILE_CONTENTS
3.c. process_system1(): ###INPUT_SYSTEM1_FILE_CONTENTS

4. Aggregates all the info from each input file (unique combination of model_name and prompt_type) into a cumulative output file in os.path.join(OUTPUT_ROOT_DIR, f'summary_all_{ENSEMBLE_NAME}') as demostrated in ###OUTPUT_CSV

5. Inserts adjustable logging/print to terminal informative feedback messages to help localize and debug at all major control flow or other statements
==========================