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
    def __init__(self, input_subdir: str, input_filename: str, output_subdir: str,
                 target_col: str = 'target', cols_drop_ls: List[str] = None):
        """Initialize preprocessor with file paths and configuration"""
        self.input_path = os.path.join(input_subdir, input_filename)
        self.output_subdir = output_subdir
        self.target_col = target_col
        self.cols_drop_ls = cols_drop_ls or []
        
        # Validate input file exists
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking variables
        self.input_stats = {}
        self.output_stats = {}
        self.transformations = []
        self.warnings = []
        
        # Load metadata 
        self.metadata = self._get_metadata_dict()
        self.logger.info("Initialized preprocessor with metadata")

    def _get_metadata_dict(self) -> Dict:
        """Define comprehensive metadata dictionary matching notebook"""
        return {
            "pubid": {
                "value": 1412,
                "renamed": "DROP",
                "type": "int64",
                "class": "demographic",
                "feature": "discrete_numeric",
            },
            "sex": {
                "value": 2,
                "renamed": "sex_recorded_in_1997",
                "type": "str",
                "class": "demographic", 
                "feature": "nominal_category",
                "labels": {
                    "Female": "female",
                    "Male": "male"
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
            "race": {
                "value": 6,
                "renamed": "race",
                "type": "str",
                "class": "demographic",
                "feature": "nominal_category",
                "labels": {
                    "american indian, eskimo or aleut": "american indian, eskimo or aleut",
                    "asian or pacific islander": "asian or pacific islander", 
                    "black": "black",
                    "hispanic": "hispanic",
                    "mixed-race": "mixed-race",
                    "white": "white"
                }
            },
            # Add other metadata mappings...
        }

    def compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute comprehensive statistics for DataFrame"""
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

    def drop_columns(self, df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
        """Drop specified columns with validation"""
        df_clean = df.copy()
        
        # Validate columns exist
        missing_cols = [col for col in cols_to_drop if col not in df_clean.columns]
        if missing_cols:
            self.warnings.append(f"Columns not found for dropping: {missing_cols}")
            cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
            
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            self.transformations.append(f"Dropped columns: {cols_to_drop}")
            self.logger.info(f"Dropped {len(cols_to_drop)} columns")
            
        return df_clean

    def convert_birthyear_to_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert birth year to age following notebook logic"""
        df_clean = df.copy()
        try:
            df_clean['age'] = 2022 - df_clean['birthyear'] 
            df_clean.drop(columns=["birthyear", "birthmonth"], inplace=True)
            self.transformations.append("Converted birth year to age")
            self.logger.info("Converted birth year to age")
        except (KeyError, TypeError) as e:
            self.logger.error(f"Age conversion failed: {str(e)}")
            df_clean['age'] = np.nan
        return df_clean

    def convert_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert height to total inches following notebook logic"""
        df_clean = df.copy()
        try:
            df_clean["height_total_inches"] = (df_clean["height_ft"] * 12) + df_clean["height_in"]
            df_clean.drop(columns=["height_ft", "height_in"], inplace=True)
            self.transformations.append("Converted height to total inches")
            self.logger.info("Converted height to inches")
        except (KeyError, TypeError) as e:
            self.logger.error(f"Height conversion failed: {str(e)}")
            df_clean["height_total_inches"] = np.nan
        return df_clean

    def clean_percentage_values(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Clean percentage values following notebook logic"""
        df_clean = df.copy()
        
        percentage_map = {
            '-4': np.nan,
            '0: 0%': 0,
            '1 TO 10: 1%-10%': 6,
            '5': 5,
            '25': 25,
            '75': 75,
            '2': 2,
            '10': 10,
            '8': 8,
            '40': 40,
            '65': 65,
            '-2': None,
            '7': 7,
            '3': 3,
            '48': 48,
            '20': 20,
            '23': 23,
            '30': 30,
            '-1': None,
            '15': 15,
            '85': 85,
            '100': 100,
            '4': 4,
            '80': 80
        }
        
        df_clean[col] = df_clean[col].map(percentage_map)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        self.transformations.append(f"Cleaned percentage values for {col}")
        self.logger.info(f"Cleaned percentages in {col}")
        return df_clean

    def convert_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert binary columns following notebook logic"""
        df_clean = df.copy()
        binary_cols = self.get_binary_columns()
        
        true_values = {
            'YES', 'Y', 'TRUE', 'T', '1', 1, 'Yes', 'True',
            True, 1.0, 'YES', 'TRUE', 'Y', 'T'
        }
        
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.upper()
                true_mask = df_clean[col].isin([str(v).upper() for v in true_values])
                df_clean[col] = true_mask
                self.transformations.append(f"Converted {col} to binary")
                
        self.logger.info(f"Converted {len(binary_cols)} binary columns")
        return df_clean

    def get_binary_columns(self) -> List[str]:
        """Get binary columns from metadata"""
        return [
            col_name for col_name, col_info in self.metadata.items()
            if col_info.get('feature') == 'binary' and col_info.get('renamed') != 'TARGET'
        ]

    def get_category_columns(self) -> List[str]:
        """Get categorical columns from metadata"""
        return [
            col_name for col_name, col_info in self.metadata.items()
            if col_info.get('feature', '').endswith('_category')
        ]

    def normalize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize categorical values following notebook logic"""
        df_clean = df.copy()
        cat_cols = self.get_category_columns()
        
        for col in cat_cols:
            if col in df.columns and 'labels' in self.metadata[col]:
                label_map = self.metadata[col]['labels']
                df_clean[col] = df_clean[col].map(label_map)
                self.transformations.append(f"Normalized categories for {col}")
                
        self.logger.info(f"Normalized {len(cat_cols)} categorical columns")
        return df_clean

    def process_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process column names following notebook logic"""
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
        
        self.transformations.extend([
            f"Dropped columns: {cols_to_drop}",
            f"Renamed columns: {rename_map}"
        ])
        
        self.logger.info(f"Processed column names - dropped {len(cols_to_drop)}, renamed {len(rename_map)}")
        return df_clean

    def write_report(self) -> None:
        """Write preprocessing report"""
        report_path = os.path.join(self.output_subdir, 'preprocessing_report.txt')
        with open(report_path, 'w') as f:
            f.write("DATA PREPROCESSING REPORT\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write("INPUT DATASET\n")
            f.write("=============\n")
            for col, stats in self.input_stats.items():
                f.write(f"\nColumn: {col}\n")
                for stat, value in stats.items():
                    f.write(f"{stat}: {value}\n")
                    
            f.write("\nOUTPUT DATASET\n")
            f.write("==============\n")
            for col, stats in self.output_stats.items():
                f.write(f"\nColumn: {col}\n")
                for stat, value in stats.items():
                    f.write(f"{stat}: {value}\n")
                    
            f.write("\nTRANSFORMATIONS\n")
            f.write("===============\n")
            for transform in self.transformations:
                f.write(f"- {transform}\n")
                
            f.write("\nWARNINGS\n")
            f.write("========\n")
            for warning in self.warnings:
                f.write(f"- {warning}\n")

    def preprocess(self) -> pd.DataFrame:
        """Main preprocessing method following notebook order exactly"""
        self.logger.info("Starting data preprocessing...")
        
        try:
            # Read input data
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Read input file with shape: {df.shape}")
            
            # Compute input statistics
            self.input_stats = self.compute_stats(df)
            
            # Follow notebook order exactly
            
            # 1. Drop initial columns
            if self.cols_drop_ls:
                df = self.drop_columns(df, self.cols_drop_ls)
            
            # 2. Convert birth year to age
            df = self.convert_birthyear_to_age(df)
            
            # 3. Convert height to inches
            df = self.convert_height(df)
            
            # 4. Clean percentage values
            for col in ['expectdeath']:
                if col in df.columns:
                    df = self.clean_percentage_values(df, col)
            
            # 5. Convert binary columns
            df = self.convert_binary_columns(df)
            
            # 6. Normalize categories
            df = self.normalize_categories(df)
            
            # 7. Process column names (rename/drop)
            df = self.process_column_names(df)
            
            # Validate target column exists
            if self.target_col not in df.columns:
                raise DataValidationError(f"Target column '{self.target_col}' not found")
                
            # Compute output statistics
            self.output_stats = self.compute_stats(df)
            
            # Save preprocessed data
            output_path = os.path.join(self.output_subdir, 'preprocessed_data.csv')
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved preprocessed data with shape: {df.shape}")
            
            # Write reports
            self.write_report()
            self.write_validation_report()
            self.logger.info("Completed preprocessing and report generation")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def write_validation_report(self) -> None:
        """Write detailed data validation report"""
        report_path = os.path.join(self.output_subdir, 'validation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("DATA VALIDATION REPORT\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # Data Quality Metrics
            f.write("DATA QUALITY METRICS\n")
            f.write("===================\n")
            
            # Missing Values Analysis
            missing_data = {col: stats['missing'] 
                          for col, stats in self.output_stats.items()}
            f.write("\nMissing Values Analysis:\n")
            for col, count in missing_data.items():
                pct = (count / len(self.output_stats)) * 100
                f.write(f"- {col}: {count} values ({pct:.1f}%)\n")
            
            # Cardinality Analysis
            f.write("\nCardinality Analysis:\n")
            for col, stats in self.output_stats.items():
                if stats['dtype'] == 'object':
                    f.write(f"- {col}: {stats['unique_values']} unique values\n")
            
            # Numeric Column Analysis
            f.write("\nNumeric Column Analysis:\n")
            for col, stats in self.output_stats.items():
                if stats['dtype'] in ['int64', 'float64']:
                    f.write(f"\n{col}:\n")
                    f.write(f"  Range: [{stats['min']:.2f} - {stats['max']:.2f}]\n")
                    if 'skew' in stats:
                        f.write(f"  Skewness: {stats['skew']:.2f}\n")
            
            # Write Warnings
            f.write("\nVALIDATION WARNINGS\n")
            f.write("==================\n")
            if self.warnings:
                for warning in self.warnings:
                    f.write(f"- {warning}\n")
            else:
                f.write("No validation warnings generated\n")
            
            # Data Quality Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("===============\n")
            
            # Missing Value Recommendations
            high_missing = [col for col, count in missing_data.items() 
                          if count/len(self.output_stats) > 0.1]
            if high_missing:
                f.write("\nMissing Value Recommendations:\n")
                for col in high_missing:
                    f.write(f"- Consider imputation or removal for {col} "
                           f"({missing_data[col]} missing values)\n")
            
            # Cardinality Recommendations
            for col, stats in self.output_stats.items():
                if (stats['dtype'] == 'object' and 
                    stats['unique_values'] > len(self.output_stats) * 0.5):
                    f.write(f"- Consider encoding or binning {col} due to high cardinality\n")
            
            # Distribution Recommendations
            for col, stats in self.output_stats.items():
                if stats['dtype'] in ['int64', 'float64']:
                    if 'skew' in stats and abs(stats['skew']) > 1:
                        f.write(f"- Consider transforming {col} to address skewness\n")

def main():
    """Main function to run preprocessing with error handling"""
    try:
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
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()