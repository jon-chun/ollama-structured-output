import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

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
        # Setup paths and configuration
        self.input_path = os.path.join(input_subdir, input_filename)
        self.output_subdir = output_subdir
        self.target_col = target_col
        self.cols_drop_ls = cols_drop_ls or []
        
        # Validate input file exists
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Setup detailed logging
        self.setup_logging()
        
        # Initialize tracking variables
        self.input_stats: Dict = {}
        self.output_stats: Dict = {}
        self.transformations: List[str] = []
        self.warnings: List[str] = []
        
        # Load metadata dictionary
        self.metadata = self._get_metadata_dict()
        self.logger.info("Initialized preprocessor with metadata")

    def setup_logging(self):
        """Setup detailed logging configuration"""
        self.logger = logging.getLogger('DataPreprocessor')
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console)
        
        # File handler
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler('logs/preprocessing.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def _get_metadata_dict(self) -> Dict:
        """
        Return comprehensive metadata dictionary with column specifications
        Contains all mappings from the notebook
        """
        return {
            "sex": {
                "value": 2,
                "renamed": "sex_recorded_in_1997",
                "type": "str",
                "class": "demographic",
                "feature": "nominal_category",
                "labels": {
                    "Female": "female",
                    "Male": "male",
                    "Unknown": "unknown"
                }
            },
            "married02": {
                "value": 8,
                "renamed": "marriage_or_cohabitation_status_in_2002", 
                "type": "str",
                "class": "demographic",
                "feature": "nominal_category",
                "labels": {
                    "Never married, not cohabiting": "never married, not cohabiting",
                    "Never married, cohabiting": "never married, cohabiting", 
                    "Married, spouse present": "married, spouse present",
                    "Married, spouse absent": "married, spouse absent",
                    "Separated, not cohabiting": "separated, not cohabiting",
                    "Separated, cohabiting": "separated, cohabiting",
                    "Divorced, not cohabiting": "divorced, not cohabiting",
                    "Divorced, cohabiting": "divorced, cohabiting"
                }
            },
            "educ02": {
                "value": 4,
                "renamed": "highest_degree_by_2002",
                "type": "str", 
                "class": "education_and_work",
                "feature": "ordinal_category",
                "labels": {
                    "Associate/Junior college (AA)": "associate/junior college (aa)",
                    "Bachelor's degree (BA, BS)": "bachelor's degree (ba, bs)",
                    "GED": "ged",
                    "High school diploma (Regular 12 year program)": "high school diploma (hs)"
                }
            },
            # Add mappings for other columns...
        }

    def validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data meets requirements
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        # Check required columns exist
        required_cols = set(self.metadata.keys())
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
            
        # Check data types
        for col, meta in self.metadata.items():
            if col in df.columns:
                expected_type = meta['type']
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    self.warnings.append(
                        f"Column {col} has type {actual_type}, expected {expected_type}")
                    
        # Check value ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].min() < 0:
                self.warnings.append(
                    f"Column {col} contains negative values")
                
        self.logger.info("Completed input data validation")
        if self.warnings:
            self.logger.warning(f"Found {len(self.warnings)} validation warnings")

    def convert_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert height from feet/inches to total inches with validation"""
        df_clean = df.copy()
        try:
            # Validate height values
            invalid_ft = df_clean[df_clean['height_ft'] > 8]['height_ft']
            invalid_in = df_clean[df_clean['height_in'] > 11]['height_in']
            
            if not invalid_ft.empty:
                self.warnings.append(f"Found {len(invalid_ft)} invalid height_ft values > 8")
            if not invalid_in.empty:
                self.warnings.append(f"Found {len(invalid_in)} invalid height_in values > 11")
            
            # Convert with clipping
            df_clean['height_ft'] = df_clean['height_ft'].clip(upper=8)
            df_clean['height_in'] = df_clean['height_in'].clip(upper=11)
            df_clean["height_total_inches"] = (df_clean["height_ft"] * 12) + df_clean["height_in"]
            
            # Validate total height
            invalid_height = df_clean[df_clean['height_total_inches'] > 96]['height_total_inches']
            if not invalid_height.empty:
                self.warnings.append(f"Found {len(invalid_height)} heights > 96 inches")
                df_clean['height_total_inches'] = df_clean['height_total_inches'].clip(upper=96)
                
            df_clean.drop(columns=["height_ft", "height_in"], inplace=True)
            self.transformations.append("Converted height to total inches with validation")
            
        except (KeyError, TypeError) as e:
            self.logger.error(f"Height conversion failed: {str(e)}")
            df_clean["height_total_inches"] = np.nan
            
        return df_clean

    def convert_birthyear_to_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert birth year to age with validation"""
        df_clean = df.copy()
        try:
            # Validate birth year range
            min_year = 1960
            max_year = 2010
            invalid_years = df_clean[
                (df_clean['birthyear'] < min_year) | 
                (df_clean['birthyear'] > max_year)
            ]['birthyear']
            
            if not invalid_years.empty:
                self.warnings.append(
                    f"Found {len(invalid_years)} birth years outside {min_year}-{max_year}")
                df_clean['birthyear'] = df_clean['birthyear'].clip(min_year, max_year)
            
            # Convert to age
            df_clean['age'] = 2022 - df_clean['birthyear']
            df_clean.drop(columns=["birthyear", "birthmonth"], inplace=True)
            self.transformations.append("Converted birth year to age with validation")
            
        except (KeyError, TypeError) as e:
            self.logger.error(f"Age conversion failed: {str(e)}")
            df_clean['age'] = np.nan
            
        return df_clean

    def normalize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize categorical columns with validation"""
        df_clean = df.copy()
        cat_cols = self.get_category_columns()
        
        for col in cat_cols:
            if col in df.columns and 'labels' in self.metadata[col]:
                # Get mapping dictionary
                label_map = self.metadata[col]['labels']
                
                # Check for unmapped values
                unmapped = set(df_clean[col].unique()) - set(label_map.keys())
                if unmapped:
                    self.warnings.append(
                        f"Found unmapped values in {col}: {unmapped}")
                    
                # Apply mapping
                df_clean[col] = df_clean[col].map(label_map)
                self.transformations.append(f"Normalized categories for {col}")
                
        return df_clean

    def clean_percentage_values(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Clean percentage values with validation"""
        df_clean = df.copy()
        
        # Define percentage mapping with validation ranges
        percentage_map = {
            '-4': np.nan,
            '0: 0%': 0,
            '1 TO 10: 1%-10%': 6,
            # Add other mappings...
        }
        
        # Convert values
        df_clean[col] = df_clean[col].map(percentage_map)
        
        # Validate percentage range
        invalid_pct = df_clean[
            (df_clean[col] < 0) | 
            (df_clean[col] > 100)
        ][col]
        
        if not invalid_pct.empty:
            self.warnings.append(
                f"Found {len(invalid_pct)} invalid percentages in {col}")
            df_clean[col] = df_clean[col].clip(0, 100)
            
        self.transformations.append(f"Cleaned percentage values for {col}")
        return df_clean

    def preprocess(self) -> pd.DataFrame:
        """Main preprocessing method with enhanced validation"""
        self.logger.info("Starting data preprocessing...")
        
        try:
            # Read and validate input
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Read input file with shape: {df.shape}")
            self.input_stats = self.compute_stats(df)
            self.validate_input_data(df)
            
            # Initial column drops
            if self.cols_drop_ls:
                df = df.drop(columns=self.cols_drop_ls)
                self.transformations.append(
                    f"Dropped initial columns: {self.cols_drop_ls}")
                self.logger.info(f"Dropped {len(self.cols_drop_ls)} initial columns")

            # Apply transformations in notebook order
            steps = [
                (self.convert_birthyear_to_age, "Convert birth year to age"),
                (self.convert_height, "Convert height to inches"),
                (self.clean_percentage_values, "Clean percentage values"),
                (self.convert_binary_columns, "Convert binary columns"),
                (self.normalize_categories, "Normalize categories"),
                (self.process_column_names, "Process column names")
            ]
            
            for step_func, step_name in steps:
                self.logger.info(f"Executing step: {step_name}")
                df = step_func(df)
                
            # Validate target column
            if self.target_col not in df.columns:
                raise DataValidationError(
                    f"Target column '{self.target_col}' not found")
                
            # Compute final stats
            self.output_stats = self.compute_stats(df)
            
            # Save outputs
            self.save_outputs(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise
            
    def save_outputs(self, df: pd.DataFrame) -> None:
        """Save preprocessed data and reports"""
        # Save preprocessed data
        output_path = os.path.join(self.output_subdir, 'preprocessed_data.csv')
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved preprocessed data with shape: {df.shape}")
        
        # Save reports
        self.write_report()
        self.write_validation_report()
        self.logger.info("Saved preprocessing reports")

    def write_validation_report(self) -> None:
        """Write detailed validation report"""
        report_path = os.path.join(self.output_subdir, 'validation_report.txt')
        with open(report_path, 'w') as f:
            f.write("DATA VALIDATION REPORT\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write("VALIDATION WARNINGS\n")
            f.write("==================\n")
            if self.warnings:
                for warning in self.warnings:
                    f.write(f"- {warning}\n")
            else:
                f.write("No validation warnings found\n")

def main():
    """Main function to run preprocessing"""
    try:
        # Configure paths
        input_subdir = os.path.join('data')
        output_subdir = os.path.join('data')
        
        # Ensure directories exist
        os.makedirs(output_subdir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Configure and run preprocessing
        preprocessor = DataPreprocessor(
            input_subdir=input_subdir,
            input_filename='vignettes_renamed_clean.csv',
            output_subdir=output_subdir,
            target_col='target',
            cols_drop_ls=['short_text_summary', 'long_text_summary']
        )
        
        preprocessed_df = preprocessor.preprocess()
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()