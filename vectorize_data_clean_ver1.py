import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import re

# Configuration constants
INPUT_SUBDIR = os.path.join('data')
INPUT_FILENAME = 'vignettes_renamed_clean.csv'
OUTPUT_SUBDIR = os.path.join('data')
OUTPUT_SUMMARY_REPORT = 'report_vectorization_summary.txt'
COLS_DROP_LS = ['short_text_summary', 'long_text_summary']
TARGET_COL = 'target'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_cleaning.log')
    ]
)

class DataReporter:
    """Class to generate summary reports for datasets"""
    
    @staticmethod
    def generate_column_stats(df: pd.DataFrame) -> Dict:
        """Generate statistical summaries for each column"""
        stats = {}
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'missing': df[col].isna().sum(),
                'unique_values': len(df[col].unique())
            }
            
            # Numeric column statistics
            if np.issubdtype(df[col].dtype, np.number):
                col_stats.update({
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q1': df[col].quantile(0.25),
                    'q3': df[col].quantile(0.75)
                })
            
            # Categorical column statistics
            else:
                col_stats.update({
                    'mode': df[col].mode().iloc[0] if not df[col].empty else None,
                    'value_counts': df[col].value_counts().to_dict()
                })
                
            stats[col] = col_stats
        return stats
    
    @staticmethod
    def write_summary_report(input_stats: Dict, output_stats: Dict, metadata: Dict, 
                           input_file: str, output_file: str, report_file: str):
        """Write comprehensive summary report"""
        with open(report_file, 'w') as f:
            f.write("Data Vectorization Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Input File: {input_file}\n")
            f.write(f"Output File: {output_file}\n\n")
            
            f.write("=== Input Dataset Summary ===\n")
            for col, stats in input_stats.items():
                f.write(f"\nColumn: {col}\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"  {stat_name}: {stat_value}\n")
                    
            f.write("\n=== Output Dataset Summary ===\n")
            for col, stats in output_stats.items():
                f.write(f"\nColumn: {col}\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"  {stat_name}: {stat_value}\n")
                    
            f.write("\n=== Metadata Summary ===\n")
            for col, meta in metadata.items():
                f.write(f"\nColumn: {col}\n")
                for meta_key, meta_value in meta.items():
                    f.write(f"  {meta_key}: {meta_value}\n")
                    
            f.write("\n=== Observations & Recommendations ===\n")
            # Add any specific observations or recommendations here
            f.write("1. Check columns with high missing value counts\n")
            f.write("2. Verify categorical encodings are as expected\n")
            f.write("3. Confirm target variable distribution is reasonable\n")

class DataCleaner:
    """Class to clean and preprocess the dataset according to specified rules"""
    
    def __init__(self, metadata_dict: Dict):
        """Initialize with metadata dictionary containing column specifications"""
        self.metadata = metadata_dict
        logging.info("Initialized DataCleaner with metadata dictionary")
        
    def get_binary_columns(self) -> List[str]:
        """Get list of binary columns from metadata"""
        binary_cols = [
            col_name for col_name, col_info in self.metadata.items()
            if col_info.get('feature') == 'binary' and col_info.get('renamed') != 'TARGET'
        ]
        logging.info(f"Identified {len(binary_cols)} binary columns")
        return binary_cols
        
    def get_category_columns(self) -> List[str]:
        """Get list of categorical columns from metadata"""
        cat_cols = [
            col_name for col_name, col_info in self.metadata.items()
            if col_info.get('feature', '').endswith('_category')
        ]
        logging.info(f"Identified {len(cat_cols)} categorical columns")
        return cat_cols

    def convert_to_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert binary columns to boolean type"""
        logging.info("Starting binary column conversion")
        df_clean = df.copy()
        binary_cols = self.get_binary_columns()
        
        true_values = {
            'YES', 'Y', 'TRUE', 'T', '1', 1, 'Yes', 'True',
            True, 1.0, 'YES', 'TRUE', 'Y', 'T'
        }
        
        for col in binary_cols:
            if col in df.columns:
                logging.info(f"Converting binary column: {col}")
                df_clean[col] = df_clean[col].astype(str).str.upper()
                true_mask = df_clean[col].isin([str(v).upper() for v in true_values])
                df_clean[col] = true_mask
                
        return df_clean

    def normalize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize categorical columns based on metadata labels"""
        logging.info("Starting categorical column normalization")
        df_clean = df.copy()
        cat_cols = self.get_category_columns()
        
        for col in cat_cols:
            if col in df.columns and 'labels' in self.metadata[col]:
                logging.info(f"Normalizing categorical column: {col}")
                label_map = self.metadata[col]['labels']
                df_clean[col] = df_clean[col].map(label_map)
                
        return df_clean

    def process_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process column names according to metadata rules"""
        logging.info("Starting column name processing")
        df_clean = df.copy()
        col_rename_map = {}
        cols_to_drop = []

        for col_name, specs in self.metadata.items():
            rename_value = specs.get('renamed', '')
            
            if rename_value == 'DROP' and col_name != TARGET_COL:
                if col_name in df_clean.columns:
                    cols_to_drop.append(col_name)
            elif rename_value == 'TARGET':
                if col_name in df_clean.columns:
                    col_rename_map[col_name] = f'y_{col_name}'
            elif rename_value:  # Not empty string
                if col_name in df_clean.columns:
                    col_rename_map[col_name] = rename_value

        logging.info(f"Dropping columns: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)
        
        logging.info(f"Renaming columns: {col_rename_map}")
        df_clean = df_clean.rename(columns=col_rename_map)
        
        return df_clean

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to clean and process the data"""
        logging.info("Starting main data cleaning process")
        df_clean = df.copy()
        
        # Drop specified columns first
        logging.info(f"Dropping specified columns: {COLS_DROP_LS}")
        df_clean = df_clean.drop(columns=[col for col in COLS_DROP_LS if col in df_clean.columns])
        
        # Convert binary columns
        df_clean = self.convert_to_binary(df_clean)
        
        # Normalize categorical columns
        df_clean = self.normalize_categories(df_clean)
        
        # Process column names (rename/drop)
        df_clean = self.process_column_names(df_clean)
        
        logging.info("Completed data cleaning process")
        return df_clean

def get_metadata_dict() -> Dict:
    """Return the metadata dictionary with column specifications"""
    # Add your complete metadata dictionary here
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
        # Add other columns...
    }

def main():
    """Main function to run the data cleaning process"""
    logging.info("Starting data cleaning program")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
    
    # Set up file paths
    input_path = os.path.join(INPUT_SUBDIR, INPUT_FILENAME)
    output_path = os.path.join(OUTPUT_SUBDIR, 'cleaned_' + INPUT_FILENAME)
    report_path = os.path.join(OUTPUT_SUBDIR, OUTPUT_SUMMARY_REPORT)
    
    try:
        # Load data
        logging.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Generate input data statistics
        logging.info("Generating input data statistics")
        input_stats = DataReporter.generate_column_stats(df)
        
        # Initialize cleaner with metadata
        metadata = get_metadata_dict()
        cleaner = DataCleaner(metadata)
        
        # Clean data
        df_clean = cleaner.clean_data(df)
        
        # Generate output data statistics
        logging.info("Generating output data statistics")
        output_stats = DataReporter.generate_column_stats(df_clean)
        
        # Write summary report
        logging.info(f"Writing summary report to {report_path}")
        DataReporter.write_summary_report(
            input_stats, output_stats, metadata,
            input_path, output_path, report_path
        )
        
        # Save cleaned data
        logging.info(f"Saving cleaned data to {output_path}")
        df_clean.to_csv(output_path, index=False)
        
        logging.info("Data cleaning program completed successfully")
        
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    main()