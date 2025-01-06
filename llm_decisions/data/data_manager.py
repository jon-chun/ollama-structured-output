# data_manager.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Any, Set

from config import Config


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
        self.df_validate: Optional[pd.DataFrame] = None
        
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
            df[self.config.data['outcome']] = df[self.config.data['outcome']].apply(normalize_value)
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
        required_columns = [self.config.data['risk_factors'], self.config.data['outcome']]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        empty_counts = df[required_columns].isna().sum()
        if empty_counts.any():
            logging.warning(f"Found empty values in columns: \n{empty_counts[empty_counts > 0]}")
        
        # Validate normalized target values
        valid_targets = {'YES', 'NO'}
        invalid_targets = set(df[self.config.data['outcome']].unique()) - valid_targets
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
            self.df = self.df.sample(frac=1.0, random_state=self.config.data["random_seed"]).reset_index(drop=True)

            # Perform stratified split to maintain class balance

            # 1) Train split
            df_train, df_rest = train_test_split(
                self.df,
                train_size=train_ratio,
                random_state=self.config.data["random_seed"],
                stratify=self.df[self.config.data["outcome"]]
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
                stratify=df_rest[self.config.data["outcome"]]
            )

            self.df_train = df_train
            self.df_validate = df_validate
            self.df_test = df_test

            logging.info(f"Loaded {len(self.df)} total samples")
            logging.info(f"Train: {len(self.df_train)} Validate: {len(self.df_validate)} Test: {len(self.df_test)}")
            logging.info(f"Target distribution in training set:\n{self.df_train[self.config.data['outcome']].value_counts()}")

            return len(self.df_train), len(self.df_test)
            
        except Exception as e:
            logging.error(f"Error loading and preparing data: {str(e)}")
            raise

    '''
    def get_batch(self, batch_size: int, dataset: str = 'train', exclude_ids: Optional[Set[int]] = None) -> List[Dict]:
        """
        Get a batch of samples from the specified dataset.
        """
        if self.df_train is None or self.df_test is None or self.df_validate is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
        
        # Select appropriate dataset
        if dataset not in ['train', 'test', 'validate']:
            raise ValueError("Dataset must be 'train', 'test', or 'validate'")
        
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
                'risk_factors': str(row[self.config.data['risk_factors']]),
                'target': str(row[self.config.data['outcome']])
            }
            batch_data.append(sample)
        
        # Log batch statistics
        batch_targets = pd.Series([d['target'] for d in batch_data])
        logging.debug(f"Batch target distribution:\n{batch_targets.value_counts()}")
        
        return batch_data
    '''

    def get_batch(
        self,
        batch_size: Optional[int],
        dataset: str = 'train',
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Dict]:
        """
        Get a batch of samples, optionally excluding specific IDs.
        
        This method provides samples from the specified dataset while ensuring we don't
        select already processed samples. It's designed to support the incremental 
        processing of data, allowing us to skip samples we've already handled.
        
        Args:
            batch_size: Number of samples to return (None for all remaining samples)
            dataset: Which dataset to sample from ('train', 'validate', 'test')
            exclude_ids: Set of IDs to exclude from selection (e.g., already processed samples)
            
        Returns:
            List of dictionaries containing sample data with keys:
            - 'id': Sample identifier
            - 'risk_factors': Risk factors text
            - 'target': Target value (YES/NO)
            
        Raises:
            RuntimeError: If data hasn't been loaded yet
            ValueError: If invalid dataset specified
        
        Example:
            >>> # Get 5 samples, excluding IDs 1, 2, and 3
            >>> batch = data_manager.get_batch(5, 'train', {1, 2, 3})
        """
        # First, verify data is loaded
        if self.df_train is None or self.df_test is None or self.df_validate is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
        
        # Validate and select appropriate dataset
        if dataset not in ['train', 'test', 'validate']:
            raise ValueError(f"Invalid dataset: {dataset}. Must be 'train', 'test', or 'validate'")
        
        df = getattr(self, f'df_{dataset}')
        
        # Start with all available samples
        available_samples = df.copy()
        
        # Filter out excluded IDs if provided
        if exclude_ids:
            original_count = len(available_samples)
            available_samples = available_samples[~available_samples['id'].isin(exclude_ids)]
            excluded_count = original_count - len(available_samples)
            logging.debug(
                f"Excluded {excluded_count} samples, {len(available_samples)} remaining in "
                f"{dataset} dataset"
            )
        
        # Handle case where no samples are available
        if len(available_samples) == 0:
            logging.warning(f"No available samples in {dataset} dataset after exclusions")
            return []
        
        # Handle batch size appropriately
        if batch_size is None:
            batch_df = available_samples
            logging.info(f"Using all {len(batch_df)} available samples from {dataset} dataset")
        else:
            # Adjust batch size if needed
            actual_batch_size = min(batch_size, len(available_samples))
            if actual_batch_size < batch_size:
                logging.warning(
                    f"Requested batch size {batch_size} reduced to {actual_batch_size} "
                    f"due to available sample count"
                )
            
            # Get random sample of rows
            batch_df = available_samples.sample(n=actual_batch_size)
            logging.info(f"Selected batch of {actual_batch_size} samples from {dataset} dataset")
        
        # Convert selected rows to list of dictionaries
        batch_data = [
            {
                'id': int(row['id']),
                'risk_factors': str(row[self.config.data['risk_factors']]),
                'target': str(row[self.config.data['outcome']])
            }
            for _, row in batch_df.iterrows()
        ]
        
        # Log distribution information for monitoring
        if batch_data:
            batch_targets = pd.Series([d['target'] for d in batch_data])
            logging.debug(
                f"Batch target distribution for {dataset}:\n"
                f"{batch_targets.value_counts().to_dict()}"
            )
        
        return batch_data

    def get_risk_factors(self, row_id: int) -> str:
        """Get risk factors text for a specific row"""
        if self.df_train is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
            
        row = self.df_train[self.df_train['id'] == row_id]
        if len(row) == 0:
            raise ValueError(f"Row ID {row_id} not found in training data")
            
        return row[self.config.data['risk_factors']].iloc[0]

    def get_actual_value(self, row_id: int) -> str:
        """Get actual target value for a specific row"""
        if self.df_train is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
            
        row = self.df_train[self.df_train['id'] == row_id]
        if len(row) == 0:
            raise ValueError(f"Row ID {row_id} not found in training data")
            
        return row[self.config.data['outcome']].iloc[0]

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the datasets."""
        if self.df is None or self.df_train is None or self.df_test is None or self.df_validate is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data first.")
            
        return {
            'dataset_sizes': {
                'total': len(self.df),
                'train': len(self.df_train),
                'validate': len(self.df_validate),
                'test': len(self.df_test)
            },
            'target_distributions': {
                'overall': self.df[self.config.data['outcome']].value_counts().to_dict(),
                'train': self.df_train[self.config.data['outcome']].value_counts().to_dict(),
                'validate': self.df_validate[self.config.data['outcome']].value_counts().to_dict(),
                'test': self.df_test[self.config.data['outcome']].value_counts().to_dict()
            },
            'feature_statistics': {
                'summary_length_mean': self.df[self.config.data['risk_factors']].str.len().mean(),
                'summary_length_std': self.df[self.config.data['risk_factors']].str.len().std(),
                'summary_length_range': [
                    self.df[self.config.data['risk_factors']].str.len().min(),
                    self.df[self.config.data['risk_factors']].str.len().max()
                ]
            }
        }