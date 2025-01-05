# data_manager.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Any

from config_ver7 import Config


class DataManager:
    """
    Manages data loading, splitting, and preprocessing for model evaluation.
    
    This class handles loading CSV data, preprocessing target values, splitting
    into train/test sets, and providing access to preprocessed data for model
    evaluation. It includes validation and normalization of target values to
    ensure compatibility with the evaluation system.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the DataManager with configuration settings.
        
        The Config object provides all necessary settings for data processing,
        including file paths, split ratios, and random seeds.
        
        Args:
            config: Configuration object containing data settings
        """
        self.config = config
        self.data_path = Path(config.data["input_file"])
        # Initialize dataframes as None until data is loaded
        self.df: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        
    def _normalize_target_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize target values to YES/NO format.
        
        This method handles various input formats and standardizes them:
        - Binary values (1/0)
        - Boolean values (True/False)
        - String variations ('yes'/'no', 'true'/'false', etc.)
        
        Args:
            df: DataFrame containing target values to normalize
            
        Returns:
            DataFrame with normalized target values in YES/NO format
            
        Raises:
            ValueError: If unrecognized target values are encountered
        """
        df = df.copy()
        
        # Define standardized value mappings
        positive_values = {1, '1', True, 'true', 'True', 'TRUE', 'YES', 'yes', 'Y', 'y'}
        negative_values = {0, '0', False, 'false', 'False', 'FALSE', 'NO', 'no', 'N', 'n'}
        
        def normalize_value(value):
            """Helper function to normalize individual values"""
            if pd.isna(value):
                return value
            
            # Handle numeric values
            if isinstance(value, (int, float)):
                value = int(value)
            
            # Convert to standard string format
            value_str = str(value).strip()
            
            # Map to standardized values
            if value in positive_values or value_str in positive_values:
                return 'YES'
            elif value in negative_values or value_str in negative_values:
                return 'NO'
            else:
                raise ValueError(f"Unrecognized target value: {value}")
        
        try:
            # Apply normalization to target column
            df['target'] = df['target'].apply(normalize_value)
            logging.info("Successfully normalized target values to YES/NO format")
        except Exception as e:
            logging.error(f"Error normalizing target values: {str(e)}")
            raise
            
        return df
        
    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the loaded data for required columns and data quality.
        
        Performs comprehensive validation including:
        - Checking for required columns
        - Identifying missing values
        - Validating target value formats
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation requirements are not met
        """
        # Verify required columns exist
        required_columns = ['short_text_summary', 'target']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        empty_counts = df[required_columns].isna().sum()
        if empty_counts.any():
            logging.warning(f"Found empty values in columns: \n{empty_counts[empty_counts > 0]}")
        
        # Validate normalized target values
        valid_targets = {'YES', 'NO'}
        invalid_targets = set(df['target'].unique()) - valid_targets
        if invalid_targets:
            raise ValueError(
                f"Invalid target values found: {invalid_targets}. "
                f"Expected values: {valid_targets}"
            )
    
    def load_and_prepare_data(self) -> Tuple[int, int]:
        """
        Load CSV data, normalize target values, and prepare train/test splits.
        
        This method performs the complete data preparation process:
        1. Loads raw data from CSV
        2. Adds sequential IDs for tracking
        3. Normalizes target values to standard format
        4. Validates data quality
        5. Creates stratified train/test split
        
        Returns:
            Tuple containing (train_size, test_size)
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If data validation fails
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        

        train_ratio = self.config.data.get("train_split", 20) / 100.0
        validate_ratio = self.config.data.get("validate_split", 20) / 100.0
        test_ratio = self.config.data.get("test_split", 60) / 100.0


        # Sanity check: they should sum up to ~1.0
        total_ratio = train_ratio + validate_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"train/validate/test ratios do not sum to 1.0 (found {total_ratio})")



        try:
            # Load data and add tracking IDs
            self.df = pd.read_csv(self.data_path)
            self.df['id'] = np.arange(len(self.df))

            # Normalize and validate target values
            self.df = self._normalize_target_values(self.df)
            self.validate_data(self.df)
            
            # Create reproducible random split
            # np.random.seed(self.config.data["random_seed"])
            # self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.df = self.df.sample(frac=1.0, random_state=self.config.data["random_seed"]).reset_index(drop=True)

            # Perform stratified split to maintain class balance

            # 1) Train split
            df_train, df_rest = train_test_split(
                self.df,
                train_size=train_ratio,
                random_state=self.config.data["random_seed"],
                stratify=self.df['target']
            )

            # 2) Validate from df_rest
            # proportion of df_rest we want for validation:
            # validate_ratio is of the entire dataset,
            # but we only have the remainder, so we compute
            val_fraction_of_rest = validate_ratio / (validate_ratio + test_ratio)

            df_validate, df_test = train_test_split(
                df_rest,
                train_size=val_fraction_of_rest,
                random_state=self.config.data["random_seed"],
                stratify=df_rest['target']
            )

            self.df_train = df_train
            self.df_validate = df_validate
            self.df_test = df_test

            logging.info(f"Loaded {len(self.df)} total samples")
            logging.info(f"Train: {len(self.df_train)} Validate: {len(self.df_validate)} Test: {len(self.df_test)}")
            logging.info(f"Target distribution in training set:\n{self.df_train['target'].value_counts()}")

            # Log data preparation results
            # logging.info(f"Loaded {len(self.df)} total samples")
            # logging.info(f"Split into {len(self.df_train)} train and {len(self.df_test)} test samples")
            # logging.info(f"Target distribution in training set: \n{self.df_train['target'].value_counts()}")
            
            return len(self.df_train), len(self.df_test)
            
        except Exception as e:
            logging.error(f"Error loading and preparing data: {str(e)}")
            raise


            
    def get_risk_factors(self, row_id: int) -> str:
        """Get risk factors text for a specific row"""
        if self.df_train is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
            
        row = self.df_train[self.df_train['id'] == row_id]
        if len(row) == 0:
            raise ValueError(f"Row ID {row_id} not found in training data")
            
        return row['short_text_summary'].iloc[0]
    
    def get_actual_value(self, row_id: int) -> str:
        """Get actual target value for a specific row"""
        if self.df_train is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
            
        row = self.df_train[self.df_train['id'] == row_id]
        if len(row) == 0:
            raise ValueError(f"Row ID {row_id} not found in training data")
            
        return row['target'].iloc[0]
    
    def get_batch(self, batch_size: int, dataset: str = 'train') -> List[Dict]:
        """
        Get a batch of samples from the specified dataset.
        
        This method returns a list of dictionaries containing all necessary information
        for each data sample in the batch. It provides efficient batch processing
        capabilities for the evaluation system.
        
        Args:
            batch_size: Number of samples to retrieve (or all samples if None)
            dataset: Which dataset to use ('train' or 'test')
            
        Returns:
            List of dictionaries, each containing:
                - id: Row identifier
                - risk_factors: Text summary of risk factors
                - target: Normalized target value (YES/NO)
                
        Raises:
            RuntimeError: If data hasn't been loaded
            ValueError: If dataset parameter is invalid
        """
        if self.df_train is None or self.df_test is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
        
        # Select appropriate dataset
        if dataset not in ['train', 'test', 'validate']:
            raise ValueError("Dataset must be 'train' or 'test' or 'validate'")
        
        # df = self.df_train if dataset == 'train' else self.df_test

        if dataset == 'train':
            df = self.df_train
        elif dataset == 'validate':
            df = self.df_validate
        elif dataset == 'test':
            df = self.df_test
        else:
            raise ValueError(f"Invalid dataset: {dataset}")


        # If batch_size is None or larger than dataset, use entire dataset
        if batch_size is None or batch_size >= len(df):
            batch_df = df
            logging.info(f"Using entire {dataset} dataset ({len(df)} samples)")
        else:
            # Get random batch
            batch_df = df.sample(n=batch_size)
            logging.info(f"Selected batch of {batch_size} samples from {dataset} dataset")
        
        # Convert batch to list of dictionaries
        batch_data = []
        for _, row in batch_df.iterrows():
            sample = {
                'id': int(row['id']),
                'risk_factors': str(row['short_text_summary']),
                'target': str(row['target'])
            }
            batch_data.append(sample)
        
        # Log batch statistics
        batch_targets = pd.Series([d['target'] for d in batch_data])
        logging.debug(f"Batch target distribution:\n{batch_targets.value_counts()}")
        
        return batch_data
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the datasets.
        
        Returns a dictionary containing dataset statistics, distributions,
        and other relevant information for monitoring the evaluation process.
        
        Returns:
            Dictionary containing:
                - dataset_sizes: Count of samples in each split
                - target_distributions: Target value distributions
                - feature_statistics: Basic statistics about features
                
        Raises:
            RuntimeError: If data hasn't been loaded
        """
        if self.df is None or self.df_train is None or self.df_test is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
            
        return {
            'dataset_sizes': {
                'total': len(self.df),
                'train': len(self.df_train),
                'test': len(self.df_test)
            },
            'target_distributions': {
                'overall': self.df['target'].value_counts().to_dict(),
                'train': self.df_train['target'].value_counts().to_dict(),
                'test': self.df_test['target'].value_counts().to_dict()
            },
            'feature_statistics': {
                'summary_length_mean': self.df['short_text_summary'].str.len().mean(),
                'summary_length_std': self.df['short_text_summary'].str.len().std(),
                'summary_length_range': [
                    self.df['short_text_summary'].str.len().min(),
                    self.df['short_text_summary'].str.len().max()
                ]
            }
        }   