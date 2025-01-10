import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

################################################################################
#                              UTILITY FUNCTIONS
################################################################################

def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for a given DataFrame, with special handling for different data types.
    All values are converted to JSON-serializable Python native types.
    
    Features:
    - Handles numeric, boolean, and categorical columns appropriately
    - For numeric columns: mean, median, std, min, max, IQR
    - For boolean columns: count of True/False values, most common value
    - For categorical columns: mode and value counts
    - Ensures all output values are JSON-serializable
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing the summary statistics with JSON-serializable values
    """
    def convert_to_native_type(value: Any) -> Any:
        """Helper function to convert NumPy/Pandas types to Python native types."""
        if pd.isna(value):
            return None
        # Handle numpy numerical types
        elif hasattr(value, 'dtype'):
            if np.issubdtype(value.dtype, np.integer):
                return int(value)
            elif np.issubdtype(value.dtype, np.floating):
                return float(value)
            elif np.issubdtype(value.dtype, np.bool_):
                return bool(value)
        # Handle pandas Timestamp
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    summary = {
        'nrows': int(len(df)),  # Convert to native int
        'ncols': int(df.shape[1]),  # Convert to native int
        'columns': {},
    }
    
    for col in df.columns:
        col_data = df[col]
        col_summary = {
            'dtype': str(col_data.dtype),
            'n_nulls': int(col_data.isna().sum()),  # Convert to native int
            'unique_values': int(col_data.nunique()),  # Convert to native int
        }
        
        # Handle different data types appropriately
        if pd.api.types.is_bool_dtype(col_data):
            # Special handling for boolean columns
            valid_data = col_data.dropna()
            if len(valid_data) > 0:
                true_count = int(valid_data.sum())  # Convert to native int
                false_count = int(len(valid_data) - true_count)  # Convert to native int
                col_summary.update({
                    'true_count': true_count,
                    'false_count': false_count,
                    'true_percentage': float(true_count / len(valid_data) * 100),  # Convert to native float
                    'mode': bool(valid_data.mode().iloc[0]) if not valid_data.empty else None  # Convert to native bool
                })
        
        elif pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            # Handle numeric (non-boolean) columns
            desc = col_data.describe()
            stats = {
                'mean': desc.get('mean'),
                'median': col_data.median() if col_data.notna().any() else None,
                'std': desc.get('std'),
                'min': desc.get('min'),
                'max': desc.get('max')
            }
            # Convert all stats to native types
            col_summary.update({
                k: convert_to_native_type(v) for k, v in stats.items()
            })
            
            # Calculate IQR only for numeric columns
            if col_data.notna().any():
                q1 = float(col_data.quantile(0.25))  # Convert to native float
                q3 = float(col_data.quantile(0.75))  # Convert to native float
                col_summary['iqr'] = float(q3 - q1)  # Convert to native float
        
        else:
            # Handle categorical columns
            value_counts = col_data.value_counts()
            if not value_counts.empty:
                # Convert value_counts to native types
                value_counts_dict = {
                    str(k): int(v)  # Convert keys to strings and values to native ints
                    for k, v in value_counts.items()
                }
                col_summary.update({
                    'mode': str(value_counts.index[0]),  # Convert to string
                    'mode_count': int(value_counts.iloc[0]),  # Convert to native int
                    'unique_categories': value_counts_dict
                })
        
        summary['columns'][col] = col_summary

    return summary


################################################################################
#                           DATA PREPROCESSOR CLASS
################################################################################

class DataPreprocessor:
    """
    A class-based pipeline to:
      1. Read in data from a parameterized file location.
      2. Apply transformations in the same order as the original notebook:
         - Drop initial columns
         - Convert birthyear to age
         - Convert height
         - Clean percentage values
         - Convert binary columns
         - Normalize categories
         - Process column names
      3. Keep a designated target column for modeling.
      4. Log process details to file and console.
      5. Save a summary report with before/after metadata and transformations.
    """

    def __init__(self,
                 input_subdir: str,
                 input_filename: str,
                 output_subdir: str,
                 target_col: str = 'target',
                 cols_drop_ls: List[str] = None):
        """
        Initialize preprocessor with enhanced file handling and logging.
        """
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Input / Output paths
        self.input_path = os.path.join(input_subdir, input_filename)
        self.output_subdir = output_subdir
        os.makedirs(self.output_subdir, exist_ok=True)
        
        # Primary output files
        self.output_data_path = os.path.join(
            self.output_subdir, f'preprocessed_data_{self.timestamp}.csv'
        )
        self.output_report_path = os.path.join(
            self.output_subdir, 'report_vectorization_summary.txt'
        )
        
        # Logging
        self.logger = None
        self.setup_logging()
        
        # Basic user config
        self.target_col = target_col
        # Ensure we always drop these columns plus any the user provided
        self.initial_drop_cols = ['short_text_summary', 'long_text_summary']
        self.cols_drop_ls = cols_drop_ls or []
        self.cols_drop_ls = list(set(self.cols_drop_ls + self.initial_drop_cols))
        
        # Tracking
        self.transformations = []
        self.warnings = []
        self.column_mappings = {}  # for rename tracking
        self.logger.info(f"Initialized DataPreprocessor with input={self.input_path}, "
                         f"output_subdir={self.output_subdir}, target_col={self.target_col}")
    
    def setup_logging(self):
        """
        Setup both file and console logging for detailed debugging.
        """
        logger = logging.getLogger('DataPreprocessor')
        logger.setLevel(logging.DEBUG)
        
        # Clear previous handlers if re-running in the same session
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Log file path
        log_file = os.path.join(self.output_subdir, f'preprocessing_{self.timestamp}.log')
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        self.logger = logger
        self.logger.info(f"Logging set up. Log file: {log_file}")

    ############################################################################
    #                          HELPER / VALIDATION
    ############################################################################

    def validate_columns(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """
        Validate that the specified columns exist in the DataFrame.
        Returns True if valid, False otherwise (and logs a warning).
        """
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            self.logger.warning(msg)
            self.warnings.append(msg)
            return False
        return True
    
    def compute_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Thin wrapper to generate summary stats using the utility function.
        """
        self.logger.info("Computing summary statistics on provided dataframe.")
        return generate_summary_statistics(df)
    
    ############################################################################
    #                           TRANSFORMATIONS
    ############################################################################

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop initial columns as well as any that user configured in self.cols_drop_ls
        (including short_text_summary, long_text_summary).
        """
        df_clean = df.copy()
        
        existing_to_drop = [c for c in self.cols_drop_ls if c in df_clean.columns]
        if existing_to_drop:
            self.logger.info(f"Dropping columns: {existing_to_drop}")
            df_clean.drop(columns=existing_to_drop, inplace=True)
            self.transformations.append(f"Dropped columns: {existing_to_drop}")
        else:
            self.logger.info("No columns to drop in drop_columns() step.")
        
        return df_clean

    def convert_birthyear_to_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert 'birthyear' -> 'age' if birthyear exists, drop 'birthyear' and 'birthmonth'.
        """
        df_clean = df.copy()
        if not self.validate_columns(df_clean, ['birthyear']):
            self.logger.warning("Skipping convert_birthyear_to_age due to missing 'birthyear'.")
            return df_clean
        
        # Basic validation
        invalid_mask = (df_clean['birthyear'] < 1900) | (df_clean['birthyear'] > 2050)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            msg = f"Found {invalid_count} invalid birthyear values outside 1900-2050 range."
            self.logger.warning(msg)
            self.warnings.append(msg)
        
        df_clean['birthyear'] = df_clean['birthyear'].clip(1900, 2050)
        current_year = 2022
        df_clean['age'] = current_year - df_clean['birthyear']
        
        # Drop original columns
        for c in ['birthyear', 'birthmonth']:
            if c in df_clean.columns:
                df_clean.drop(columns=[c], inplace=True)
                self.transformations.append(f"Dropped column: {c} after birthyear->age conversion.")
        
        self.logger.info("Converted 'birthyear' to 'age' and dropped birthyear/birthmonth.")
        self.transformations.append("Converted 'birthyear' to 'age'.")
        return df_clean

    def convert_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert height_ft, height_in -> height_total_inches.
        Drop height_ft, height_in afterwards.
        """
        df_clean = df.copy()
        if not self.validate_columns(df_clean, ['height_ft', 'height_in']):
            self.logger.warning("Skipping convert_height due to missing columns.")
            return df_clean
        
        # Clip to valid ranges
        df_clean['height_ft'] = df_clean['height_ft'].clip(lower=0, upper=8)
        df_clean['height_in'] = df_clean['height_in'].clip(lower=0, upper=11)
        
        df_clean['height_total_inches'] = (
            df_clean['height_ft'] * 12 + df_clean['height_in']
        )
        
        # Drop original
        df_clean.drop(columns=['height_ft', 'height_in'], inplace=True)
        self.transformations.append("Converted [height_ft, height_in] -> height_total_inches and dropped originals.")
        self.logger.info("Converted height_ft/height_in to height_total_inches.")
        return df_clean

    def clean_percentage_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Example cleaning of expectdeath or any other % columns.
        If none exist, we skip.  If your data has a column like 'expectdeath', illustrate cleaning.
        (In the original notebook, 'expectdeath' was mapped from strings.)
        """
        df_clean = df.copy()
        if 'expectdeath' not in df_clean.columns:
            self.logger.info("No 'expectdeath' column found. Skipping clean_percentage_values.")
            return df_clean
        
        self.logger.info("Cleaning 'expectdeath' column by converting string codes to numeric or NaN.")
        # Example approach: If your dataset has specific codes, you can map them here
        # Just do a basic numeric conversion demonstration:
        def safe_to_int(val):
            try:
                return int(val)
            except:
                return np.nan
        
        df_clean['expectdeath'] = df_clean['expectdeath'].apply(safe_to_int)
        self.transformations.append("Converted 'expectdeath' from string-coded to numeric (NaN on failure).")
        return df_clean

    def convert_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns containing 'True/False', 'Yes/No', '1/0', etc. to boolean dtype.
        In the original notebook, this logic was done for multiple columns, e.g. 'arrestedafter2002', etc.
        """
        df_clean = df.copy()
        
        # Identify possible binary columns (heuristic or explicit). For demonstration, let's do a simple approach:
        candidate_cols = [c for c in df_clean.columns if df_clean[c].dropna().nunique() <= 2]
        
        for col in candidate_cols:
            # Attempt to parse as bool if the unique values are something like {True, False}, {1,0}, etc.
            unique_vals = set(df_clean[col].dropna().unique())
            # string them and uppercase for easier matching
            unique_str = {str(u).strip().upper() for u in unique_vals}
            
            # Allowed "true" tokens
            true_tokens = {'TRUE', 'T', 'YES', '1', '1.0'}
            # Allowed "false" tokens
            false_tokens = {'FALSE', 'F', 'NO', '0', '0.0'}
            
            # If all unique values are in true_tokens union false_tokens -> convert
            if unique_str.issubset(true_tokens.union(false_tokens)):
                self.logger.info(f"Converting column '{col}' to boolean.")
                def to_bool(x):
                    if str(x).strip().upper() in true_tokens:
                        return True
                    elif str(x).strip().upper() in false_tokens:
                        return False
                    else:
                        return np.nan
                df_clean[col] = df_clean[col].apply(to_bool)
                df_clean[col] = df_clean[col].astype('boolean')
                self.transformations.append(f"Converted column '{col}' to boolean.")
                
        return df_clean

    def normalize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize or rename category labels. In the original notebook, we used
        label dictionaries. Here we illustrate with a simple pass or fixed maps.
        """
        df_clean = df.copy()
        
        # Example: Suppose 'sex_recorded_in_1997' has 'Male'/'Female' -> 'male'/'female'
        if 'sex_recorded_in_1997' in df_clean.columns:
            self.logger.info("Normalizing 'sex_recorded_in_1997' category labels.")
            df_clean['sex_recorded_in_1997'] = df_clean['sex_recorded_in_1997'].replace(
                {
                    'Male': 'male',
                    'male': 'male',
                    'Female': 'female',
                    'female': 'female',
                }
            )
            self.transformations.append("Normalized sex_recorded_in_1997 to {male, female}.")
        
        # Example: if you'd like to unify race or other columns, you could do so here.
        return df_clean

    def process_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Example final rename for the target column or others. 
        If 'arrestedafter2002' is the actual target, we rename it to 'target'.
        """
        df_clean = df.copy()
        
        # If 'arrestedafter2002' is our target in the data, rename to 'target'
        if 'arrestedafter2002' in df_clean.columns and (self.target_col == 'target'):
            self.logger.info("Renaming 'arrestedafter2002' to 'target' for final dataset.")
            df_clean.rename(columns={'arrestedafter2002': 'target'}, inplace=True)
            self.column_mappings['arrestedafter2002'] = 'target'
            self.transformations.append("Renamed arrestedafter2002 -> target.")
        
        # Example: If the user wants to keep 'target' but it doesn't exist, log a warning
        if self.target_col not in df_clean.columns:
            msg = (f"WARNING: The designated target_col='{self.target_col}' is not present "
                   f"in the final DataFrame. Please confirm your pipeline.")
            self.logger.warning(msg)
            self.warnings.append(msg)
        
        return df_clean

    ############################################################################
    #                        MAIN PREPROCESSING PIPELINE
    ############################################################################

    def preprocess(self) -> pd.DataFrame:
        """
        Main pipeline:
          1. Read CSV
          2. Generate stats for input file
          3. Stepwise transformations
          4. Generate stats for output
          5. Write summary report
          6. Return final DataFrame
        """
        self.logger.info("Starting data preprocessing pipeline...")
        
        # 1. Read CSV
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        df = pd.read_csv(self.input_path)
        self.logger.info(f"Loaded input DataFrame with shape: {df.shape}")
        
        # 2. Stats for input
        self.logger.info("Computing input file statistics...")
        self.input_stats = self.compute_stats(df)
        
        # 3. Apply transformations in order
        #    (Mirroring original notebook steps)
        steps = [
            (self.drop_columns,                 "Drop initial columns"),
            (self.convert_birthyear_to_age,     "Convert birthyear->age"),
            (self.convert_height,               "Convert height"),
            (self.clean_percentage_values,      "Clean % columns"),
            (self.convert_binary_columns,       "Convert binary columns"),
            (self.normalize_categories,         "Normalize category labels"),
            (self.process_column_names,         "Process column names")
        ]
        
        for func, desc in steps:
            self.logger.info(f"=== Transformation: {desc} ===")
            df = func(df)
            self.logger.info(f"DataFrame shape after '{desc}': {df.shape}")
        
        # 4. Stats for output
        self.logger.info("Computing output file statistics...")
        self.output_stats = self.compute_stats(df)
        
        # 5. Save final CSV
        df.to_csv(self.output_data_path, index=False)
        self.logger.info(f"Saved preprocessed data to: {self.output_data_path}")

        # 6. Write summary
        self.write_summary_report()
        self.logger.info(f"Saved summary report to: {self.output_report_path}")
        
        # 7. Return final DataFrame
        self.logger.info("Preprocessing pipeline completed successfully.")
        return df

    ############################################################################
    #                          REPORTING
    ############################################################################

    def write_summary_report(self) -> None:
        """
        Write an OUTPUT_SUMMARY_REPORT = 'report_vectorization_summary.txt'
        under the output_subdir. Summarize:
          a) metadata & summary stats of input
          b) metadata & summary stats of output
          c) list of transformations
          d) warnings/recommendations
        """
        with open(self.output_report_path, 'w', encoding='utf-8') as f:
            f.write("============================================================\n")
            f.write("           PREPROCESSING SUMMARY REPORT\n")
            f.write("============================================================\n\n")
            
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Input File: {self.input_path}\n")
            f.write(f"Output File: {self.output_data_path}\n\n")
            
            # (a) Input stats
            f.write("------------------------------------------------------------\n")
            f.write("INPUT FILE METADATA & SUMMARY STATS\n")
            f.write("------------------------------------------------------------\n")
            json.dump(self.input_stats, f, indent=2)
            f.write("\n\n")
            
            # (b) Output stats
            f.write("------------------------------------------------------------\n")
            f.write("OUTPUT FILE METADATA & SUMMARY STATS\n")
            f.write("------------------------------------------------------------\n")
            json.dump(self.output_stats, f, indent=2)
            f.write("\n\n")
            
            # (c) Transformations
            f.write("------------------------------------------------------------\n")
            f.write("TRANSFORMATIONS APPLIED (in order)\n")
            f.write("------------------------------------------------------------\n")
            for i, t in enumerate(self.transformations, start=1):
                f.write(f"{i}. {t}\n")
            f.write("\n")
            
            # (d) Warnings / Observations
            f.write("------------------------------------------------------------\n")
            f.write("OBSERVATIONS / WARNINGS / RECOMMENDATIONS\n")
            f.write("------------------------------------------------------------\n")
            if not self.warnings:
                f.write("No warnings.\n")
            else:
                for i, w in enumerate(self.warnings, start=1):
                    f.write(f"{i}. {w}\n")
            f.write("\n")


################################################################################
#                                 MAIN
################################################################################

def main():
    """
    Main function with example usage. Adjust as needed for your environment.
    1) Parameterize input_subdir='data', input_filename='vignettes_renamed_clean.csv'
    2) Output_subdir='data'
    3) target_col='target'
    4) Provide any columns to drop beyond the default short_text_summary/long_text_summary
    """
    # Example user config
    input_subdir = os.path.join('data')
    input_filename = 'vignettes_renamed_clean.csv'
    output_subdir = os.path.join('data')
    
    # Instantiate the preprocessor
    preprocessor = DataPreprocessor(
        input_subdir=input_subdir,
        input_filename=input_filename,
        output_subdir=output_subdir,
        target_col='target',
        cols_drop_ls=[]  # We already drop short_text_summary & long_text_summary by default
    )
    
    # Run the pipeline
    final_df = preprocessor.preprocess()
    
    print("\nPreprocessing completed! See logs and summary report for details.")


if __name__ == "__main__":
    main()
