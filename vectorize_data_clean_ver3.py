import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path

class DataPreprocessor:
    """Class to preprocess data according to specified metadata rules"""
    
    def __init__(self, input_subdir: str, input_filename: str, output_subdir: str,
                 target_col: str = 'target', cols_drop_ls: List[str] = None):
        """Initialize preprocessor with file paths and configuration"""
        self.input_path = os.path.join(input_subdir, input_filename)
        self.output_subdir = output_subdir
        self.target_col = target_col
        self.cols_drop_ls = cols_drop_ls or []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging
        
        # Initialize metadata 
        self.metadata = self.get_metadata_dict()
        
        # Input/output stats for report
        self.input_stats = {}
        self.output_stats = {}
        self.transformations = []

    def log_step(self, message: str):
        """Log preprocessing step with timestamp"""
        self.logger.info(message)

    def get_binary_columns(self) -> List[str]:
        """Get list of binary columns from metadata"""
        return [
            col_name for col_name, col_info in self.metadata.items()
            if col_info.get('feature') == 'binary' and col_info.get('renamed') != 'TARGET'
        ]

    def convert_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert specified columns to binary type"""
        df_clean = df.copy()
        binary_cols = self.get_binary_columns()
        
        true_values = {
            'YES', 'Y', 'TRUE', 'T', '1', 1, 'Yes', 'True',
            True, 1.0, 'YES', 'TRUE', 'Y', 'T'
        }
        
        for col in binary_cols:
            if col in df.columns:
                df_clean[col] = df_clean[col].astype(str).str.upper()
                true_mask = df_clean[col].isin([str(v).upper() for v in true_values])
                df_clean[col] = true_mask
                self.log_step(f"Converted '{col}' to binary")
                
        return df_clean

    def get_category_columns(self) -> List[str]:
        """Get list of categorical columns from metadata"""
        return [
            col_name for col_name, col_info in self.metadata.items()
            if col_info.get('feature', '').endswith('_category')
        ]

    def normalize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize categorical columns based on metadata labels"""
        df_clean = df.copy()
        cat_cols = self.get_category_columns()
        
        for col in cat_cols:
            if col in df.columns and 'labels' in self.metadata[col]:
                label_map = self.metadata[col]['labels']
                df_clean[col] = df_clean[col].map(label_map)
                self.log_step(f"Normalized categories for '{col}'")
                
        return df_clean

    def convert_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert height from feet/inches to total inches"""
        df_clean = df.copy()
        try:
            df_clean["height_total_inches"] = (df_clean["height_ft"] * 12) + df_clean["height_in"]
            df_clean.drop(columns=["height_ft", "height_in"], inplace=True)
            self.log_step("Converted height to total inches")
        except (KeyError, TypeError):
            df_clean["height_total_inches"] = np.nan
            self.log_step("Failed to convert height, setting to NaN")
        return df_clean

    def convert_birthyear_to_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert birth year to age"""
        df_clean = df.copy()
        try:
            df_clean['age'] = 2022 - df_clean['birthyear']
            df_clean.drop(columns=["birthyear", "birthmonth"], inplace=True)
            self.log_step("Converted birth year to age")
        except (KeyError, TypeError):
            df_clean['age'] = np.nan
            self.log_step("Failed to convert birth year, setting age to NaN")
        return df_clean

    def clean_percentage_values(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Clean percentage values in specified column"""
        df_clean = df.copy()
        
        # Define percentage mapping
        percentage_map = {
            '-4': np.nan,
            '0: 0%': 0,
            '1 TO 10: 1%-10%': 6,
            # Add other mappings as needed
        }
        
        df_clean[col] = df_clean[col].map(percentage_map)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        self.log_step(f"Cleaned percentage values for '{col}'")
        return df_clean

    def process_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process column names according to metadata rules"""
        df_clean = df.copy()
        rename_map = {}
        cols_to_drop = []

        for col_name, specs in self.metadata.items():
            rename_value = specs.get('renamed', '')
            
            if rename_value == 'DROP':
                if col_name in df_clean.columns:
                    cols_to_drop.append(col_name)
            elif rename_value == 'TARGET':
                if col_name in df_clean.columns:
                    rename_map[col_name] = f'y_{col_name}'
            elif rename_value:
                if col_name in df_clean.columns:
                    rename_map[col_name] = rename_value

        df_clean = df_clean.drop(columns=cols_to_drop)
        df_clean = df_clean.rename(columns=rename_map)
        
        self.log_step(f"Processed column names - dropped {len(cols_to_drop)}, renamed {len(rename_map)}")
        return df_clean

    def preprocess(self) -> pd.DataFrame:
        """Main preprocessing method - follows notebook order exactly"""
        self.log_step("Starting data preprocessing...")
        
        # Read input data
        df = pd.read_csv(self.input_path)
        self.log_step(f"Read input file with shape: {df.shape}")
        self.input_stats = self.compute_stats(df)
        
        # Drop initial columns (e.g. text summaries)
        if self.cols_drop_ls:
            df = df.drop(columns=self.cols_drop_ls)
            self.transformations.append(f"Dropped initial columns: {self.cols_drop_ls}")
            self.log_step(f"Dropped {len(self.cols_drop_ls)} initial columns")

        # Convert birthyear to age
        df = self.convert_birthyear_to_age(df)
        
        # Convert height to total inches
        df = self.convert_height(df)
        
        # Clean percentage columns
        for col in ['expectdeath']:
            if col in df.columns:
                df = self.clean_percentage_values(df, col)
        
        # Convert binary columns
        df = self.convert_binary_columns(df)
        
        # Normalize categorical columns
        df = self.normalize_categories(df)
        
        # Process column renames and drops
        df = self.process_column_names(df)
        
        # Validate target column exists
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found after preprocessing")
            
        # Compute output stats
        self.output_stats = self.compute_stats(df)
        
        # Save preprocessed data
        output_path = os.path.join(self.output_subdir, 'preprocessed_data.csv')
        df.to_csv(output_path, index=False)
        self.log_step(f"Saved preprocessed data with shape: {df.shape}")
        
        # Write summary report
        report_path = os.path.join(self.output_subdir, 'report_vectorization_summary.txt')
        self.write_report(report_path)
        self.log_step("Wrote preprocessing summary report")
        
        return df

    def compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics for DataFrame columns"""
        stats = {}
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'missing': df[col].isnull().sum(),
                'unique_values': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_stats.update({
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                })
            elif df[col].dtype == 'object':
                col_stats['value_counts'] = df[col].value_counts().to_dict()
                
            stats[col] = col_stats
            
        return stats

    def write_report(self, output_path: str):
        """Write preprocessing summary report"""
        with open(output_path, 'w') as f:
            f.write("DATA PREPROCESSING SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write("INPUT DATASET SUMMARY\n")
            f.write("=====================\n")
            for col, stats in self.input_stats.items():
                f.write(f"\nColumn: {col}\n")
                for stat, value in stats.items():
                    f.write(f"{stat}: {value}\n")
                    
            f.write("\nOUTPUT DATASET SUMMARY\n")
            f.write("======================\n")
            for col, stats in self.output_stats.items():
                f.write(f"\nColumn: {col}\n")
                for stat, value in stats.items():
                    f.write(f"{stat}: {value}\n")
                    
            f.write("\nTRANSFORMATIONS APPLIED\n")
            f.write("======================\n")
            for transform in self.transformations:
                f.write(f"- {transform}\n")

def main():
    """Main function to run preprocessing"""
    # Configure paths
    input_subdir = os.path.join('data')
    output_subdir = os.path.join('data')
    
    # Ensure output directory exists
    os.makedirs(output_subdir, exist_ok=True)
    
    # Configure preprocessing
    preprocessor = DataPreprocessor(
        input_subdir=input_subdir,
        input_filename='vignettes_renamed_clean.csv',
        output_subdir=output_subdir,
        target_col='target',
        cols_drop_ls=['short_text_summary', 'long_text_summary']
    )
    
    # Run preprocessing
    preprocessed_df = preprocessor.preprocess()
    
if __name__ == "__main__":
    main()