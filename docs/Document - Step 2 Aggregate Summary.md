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