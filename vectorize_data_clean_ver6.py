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
from datetime import datetime

class DataPreprocessor:
    def __init__(self, input_subdir: str, input_filename: str, output_subdir: str,
                 target_col: str = 'target', cols_drop_ls: List[str] = None):
        """Initialize preprocessor with enhanced file handling"""
        # Setup paths with clear naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.input_path = os.path.join(input_subdir, input_filename)
        self.output_dir = os.path.join(output_subdir, f'preprocessing_output_{timestamp}')
        self.output_data_path = os.path.join(self.output_dir, 'preprocessed_data.csv')
        self.output_report_path = os.path.join(self.output_dir, 'preprocessing_report.txt')
        self.output_validation_path = os.path.join(self.output_dir, 'validation_report.txt')
        self.output_plots_dir = os.path.join(self.output_dir, 'plots')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_plots_dir, exist_ok=True)
        
        self.target_col = target_col
        self.cols_drop_ls = cols_drop_ls or []
        
        # Setup enhanced logging
        self.setup_logging()
        
        # Initialize tracking
        self.input_stats = {}
        self.output_stats = {}
        self.transformations = []
        self.warnings = []
        self.column_mappings = {}
        
        # Load metadata
        self.metadata = self._get_metadata_dict()
        self.logger.info(f"Initialized preprocessor - outputs will be saved to: {self.output_dir}")

    def setup_logging(self):
        """Setup detailed logging with file output"""
        self.logger = logging.getLogger('DataPreprocessor')
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler with timestamp
        log_file = os.path.join(log_dir, f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console)
        
        self.logger.info(f"Logging to: {log_file}")

    def validate_columns(self, df: pd.DataFrame, required_cols: List[str]) -> None:
        """Validate required columns exist before transformation"""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            self.logger.warning(msg)
            self.warnings.append(msg)
            return False
        return True

    def generate_visualizations(self, df: pd.DataFrame, stage: str = 'input') -> None:
        """Generate and save data visualizations"""
        plt.style.use('seaborn')
        
        # 1. Missing Values Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title(f'Missing Values Heatmap ({stage} data)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_plots_dir, f'missing_values_{stage}.png'))
        plt.close()

        # 2. Numeric Distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(
                ceil(len(numeric_cols)/3), 3, 
                figsize=(15, 5*ceil(len(numeric_cols)/3))
            )
            axes = axes.flatten()
            
            for idx, col in enumerate(numeric_cols):
                sns.histplot(data=df, x=col, ax=axes[idx])
                axes[idx].set_title(f'{col} Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_plots_dir, f'numeric_distributions_{stage}.png'))
            plt.close()

        # 3. Categorical Value Counts
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                plt.figure(figsize=(10, 5))
                df[col].value_counts().plot(kind='bar')
                plt.title(f'{col} Value Counts')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_plots_dir, 
                                       f'categorical_counts_{col}_{stage}.png'))
                plt.close()

    def convert_birthyear_to_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert birth year to age with enhanced validation"""
        df_clean = df.copy()
        
        # Validate required columns
        if not self.validate_columns(df_clean, ['birthyear']):
            self.logger.warning("Skipping age conversion due to missing columns")
            return df_clean
            
        try:
            # Validate birth year values
            valid_years = (df_clean['birthyear'] >= 1960) & (df_clean['birthyear'] <= 2010)
            invalid_count = (~valid_years).sum()
            
            if invalid_count > 0:
                msg = f"Found {invalid_count} invalid birth years"
                self.logger.warning(msg)
                self.warnings.append(msg)
                
            # Convert to age with clipping
            df_clean['birthyear'] = df_clean['birthyear'].clip(1960, 2010)
            df_clean['age'] = 2022 - df_clean['birthyear']
            
            # Drop original columns
            df_clean.drop(columns=['birthyear', 'birthmonth'], inplace=True, errors='ignore')
            
            self.transformations.append(
                f"Converted birth year to age (invalid years: {invalid_count})")
            self.logger.info(f"Converted birth year to age")
            
        except Exception as e:
            self.logger.error(f"Age conversion failed: {str(e)}")
            df_clean['age'] = np.nan
            
        return df_clean

    def convert_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert height with enhanced validation"""
        df_clean = df.copy()
        
        # Validate required columns
        if not self.validate_columns(df_clean, ['height_ft', 'height_in']):
            self.logger.warning("Skipping height conversion due to missing columns")
            return df_clean
            
        try:
            # Validate height ranges
            invalid_ft = df_clean[df_clean['height_ft'] > 8]['height_ft']
            invalid_in = df_clean[df_clean['height_in'] > 11]['height_in']
            
            if len(invalid_ft) > 0:
                msg = f"Found {len(invalid_ft)} heights > 8 feet"
                self.logger.warning(msg)
                self.warnings.append(msg)
                
            if len(invalid_in) > 0:
                msg = f"Found {len(invalid_in)} inches > 11"
                self.logger.warning(msg)
                self.warnings.append(msg)
                
            # Convert with clipping
            df_clean['height_ft'] = df_clean['height_ft'].clip(0, 8)
            df_clean['height_in'] = df_clean['height_in'].clip(0, 11)
            df_clean["height_total_inches"] = (df_clean["height_ft"] * 12) + df_clean["height_in"]
            
            # Validate total height
            invalid_total = df_clean[df_clean['height_total_inches'] > 96]
            if len(invalid_total) > 0:
                msg = f"Found {len(invalid_total)} total heights > 96 inches"
                self.logger.warning(msg)
                self.warnings.append(msg)
                df_clean['height_total_inches'] = df_clean['height_total_inches'].clip(0, 96)
                
            # Drop original columns
            df_clean.drop(columns=['height_ft', 'height_in'], inplace=True)
            
            self.transformations.append(
                "Converted height to total inches with validation")
            self.logger.info("Converted height measurements")
            
        except Exception as e:
            self.logger.error(f"Height conversion failed: {str(e)}")
            df_clean["height_total_inches"] = np.nan
            
        return df_clean

    def write_detailed_report(self) -> None:
        """Write comprehensive report with visualizations"""
        with open(self.output_report_path, 'w') as f:
            f.write("DETAILED DATA PREPROCESSING REPORT\n")
            f.write("================================\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # File Information
            f.write("FILE INFORMATION\n")
            f.write("================\n")
            f.write(f"Input file: {self.input_path}\n")
            f.write(f"Output file: {self.output_data_path}\n")
            f.write(f"Reports directory: {self.output_dir}\n")
            f.write(f"Visualizations directory: {self.output_plots_dir}\n\n")
            
            # Data Transformations
            f.write("TRANSFORMATIONS APPLIED\n")
            f.write("=====================\n")
            for i, transform in enumerate(self.transformations, 1):
                f.write(f"{i}. {transform}\n")
            
            # Column Changes
            f.write("\nCOLUMN CHANGES\n")
            f.write("==============\n")
            f.write("Dropped Columns:\n")
            for col in self.cols_drop_ls:
                f.write(f"- {col}\n")
            
            f.write("\nRenamed Columns:\n")
            for old_name, new_name in self.column_mappings.items():
                f.write(f"- {old_name} → {new_name}\n")
            
            # Data Quality Issues
            f.write("\nDATA QUALITY ISSUES\n")
            f.write("==================\n")
            for warning in self.warnings:
                f.write(f"- {warning}\n")
            
            # Generated Visualizations
            f.write("\nGENERATED VISUALIZATIONS\n")
            f.write("=======================\n")
            for plot_file in os.listdir(self.output_plots_dir):
                f.write(f"- {plot_file}\n")

    def preprocess(self) -> pd.DataFrame:
        """Main preprocessing method with enhanced reporting"""
        self.logger.info("Starting data preprocessing...")
        
        try:
            # Read input data
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Read input file with shape: {df.shape}")
            
            # Generate input visualizations
            self.generate_visualizations(df, 'input')
            self.input_stats = self.compute_stats(df)
            
            # Follow notebook order with validation
            processing_steps = [
                (self.drop_columns, "Drop initial columns"),
                (self.convert_birthyear_to_age, "Convert birth year to age"),
                (self.convert_height, "Convert height to inches"),
                (self.clean_percentage_values, "Clean percentage values"),
                (self.convert_binary_columns, "Convert binary columns"),
                (self.normalize_categories, "Normalize categories"),
                (self.process_column_names, "Process column names")
            ]
            
            for step_func, step_name in processing_steps:
                self.logger.info(f"Executing: {step_name}")
                df = step_func(df)
                # Generate intermediate visualizations for key steps
                if step_name in ["Convert birth year to age", "Clean percentage values"]:
                    self.generate_visualizations(df, f'after_{step_name.lower().replace(" ", "_")}')
            
            # Validate target column
            if self.target_col not in df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found")
            
            # Generate output visualizations and stats
            self.generate_visualizations(df, 'output')
            self.output_stats = self.compute_stats(df)
            
            # Save outputs
            df.to_csv(self.output_data_path, index=False)
            self.write_detailed_report()
            self.logger.info(f"Saved preprocessed data to: {self.output_data_path}")
            self.logger.info(f"Saved detailed report to: {self.output_report_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise


def create_validation_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Creates a comprehensive set of validation plots for data quality assessment.
    
    This function generates multiple types of visualizations to help understand:
    - Data distributions and patterns
    - Missing value patterns
    - Feature correlations
    - Category balances
    - Key metric distributions
    
    Args:
        df: Input DataFrame to analyze
        output_dir: Directory to save generated plots
    """
    # Create a dedicated plots directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = os.path.join(output_dir, f'validation_plots_{timestamp}')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set consistent style for visual coherence
    plt.style.use('seaborn')
    
    # Generate correlation matrix for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, 
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300)
        plt.close()
    
    # Analyze and visualize missing data patterns
    plt.figure(figsize=(12, 6))
    missing_data = df.isnull().sum()/len(df) * 100
    missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
    
    if not missing_data.empty:
        ax = missing_data.plot(kind='barh')
        plt.title('Missing Data Analysis')
        plt.xlabel('Percentage Missing (%)')
        ax.bar_label(ax.containers[0], fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'missing_values.png'), dpi=300)
    plt.close()
    
    # Create distribution plots for key numeric metrics
    key_metrics = ['age', 'height_total_inches', 'weight_in_lbs']
    for metric in key_metrics:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=metric, kde=True, bins=30)
            plt.title(f'Distribution of {metric}')
            
            # Add statistical annotations
            mean_val = df[metric].mean()
            median_val = df[metric].median()
            plt.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.1f}')
            plt.axvline(median_val, color='green', linestyle='--', 
                       label=f'Median: {median_val:.1f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{metric}_distribution.png'), 
                       dpi=300)
            plt.close()
    
    # Analyze categorical variable distributions
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        # Skip columns with too many unique values
        if df[col].nunique() <= 20:  # Limit to prevent overcrowded plots
            plt.figure(figsize=(10, 6))
            value_counts = df[col].value_counts()
            ax = value_counts.plot(kind='bar')
            plt.title(f'Value Distribution for {col}')
            plt.xticks(rotation=45, ha='right')
            
            # Add percentage labels
            total = len(df)
            percentages = [(count/total)*100 for count in value_counts]
            ax.bar_label(ax.containers[0], 
                        labels=[f'{p:.1f}%' for p in percentages],
                        rotation=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{col}_distribution.png'), 
                       dpi=300)
            plt.close()

def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    "


def main():
    """Main function with enhanced output management"""
    try:
        # Configure paths
        base_dir = os.getcwd()
        input_subdir = os.path.join(base_dir, 'data')
        output_subdir = os.path.join(base_dir, 'outputs')
        
        # Ensure directories exist
        os.makedirs(input_subdir, exist_ok=True)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Initialize and run preprocessor
        preprocessor = DataPreprocessor(
            input_subdir=input_subdir,
            input_filename='vignettes_renamed_clean.csv',
            output_subdir=output_subdir,
            target_col='target',
            cols_drop_ls=['short_text_summary', 'long_text_summary']
        )
        
        # Run preprocessing
        preprocessed_df = preprocessor.preprocess()
        
        print("\nPreprocessing completed successfully!")
