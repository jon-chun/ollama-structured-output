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
        """
        Initialize preprocessor with file paths and configuration
        
        Args:
            input_subdir: Input directory path
            input_filename: Input CSV filename  
            output_subdir: Output directory path
            target_col: Name of target column
            cols_drop_ls: List of columns to drop initially
        """
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

    def get_metadata_dict(self) -> Dict:
        """Return metadata dictionary with column specifications"""
        return {
            "sex": {
                "renamed": "sex_recorded_in_1997",
                "type": "str", 
                "feature": "nominal_category",
                "labels": {
                    "Female": "female",
                    "Male": "male"
                }
            },
            "married02": {
                "renamed": "marriage_or_cohabitation_status_in_2002",
                "type": "str",
                "feature": "nominal_category",
                "labels": {
                    "Never married, not cohabiting": "never married, not cohabiting",
                    "Never married, cohabiting": "never married, cohabiting",
                    "Married, spouse present": "married, spouse present",
                    # Add other marriage status mappings...
                }
            },
            # Add other column metadata...
        }

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

    def preprocess(self):
        """Main preprocessing method"""
        self.log_step("Starting data preprocessing...")
        
        # Read input data
        df = pd.read_csv(self.input_path)
        self.log_step(f"Read input file with shape: {df.shape}")
        
        # Compute input stats
        self.input_stats = self.compute_stats(df)
        
        # Initial column drops
        if self.cols_drop_ls:
            df = df.drop(columns=self.cols_drop_ls)
            self.transformations.append(f"Dropped initial columns: {self.cols_drop_ls}")
            self.log_step(f"Dropped {len(self.cols_drop_ls)} initial columns")
            
        # Convert binary columns
        binary_cols = [col for col, meta in self.metadata.items() 
                      if meta.get('feature') == 'binary']
        if binary_cols:
            df = self.convert_binary_columns(df, binary_cols)
            self.transformations.append(f"Converted binary columns: {binary_cols}")
            
        # Normalize categorical columns
        cat_cols = [col for col, meta in self.metadata.items()
                   if meta.get('feature', '').endswith('_category')]
        if cat_cols:
            df = self.normalize_categories(df, cat_cols)
            self.transformations.append(f"Normalized categorical columns: {cat_cols}")
            
        # Process column renames
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