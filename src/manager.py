# data_manager.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple

class DataManager:
    """Manages data loading, splitting, and preprocessing for model evaluation"""
    def __init__(self, config: 'Config'):
        self.config = config
        self.data_path = Path(config.data["input_file"])
        self.df = None
        self.df_train = None
        self.df_test = None
        
    def load_and_prepare_data(self) -> None:
        """Load CSV data and prepare train/test splits"""
        # Read CSV and add sequence ID
        self.df = pd.read_csv(self.data_path)
        self.df['id'] = np.arange(len(self.df))
        
        # Set random seed and shuffle
        np.random.seed(self.config.data["random_seed"])
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        # Split into train/test sets
        train_size = self.config.data["train_split"]
        self.df_train, self.df_test = train_test_split(
            self.df, 
            train_size=train_size,
            random_state=self.config.data["random_seed"]
        )
        
        logging.info(f"Loaded {len(self.df)} total samples")
        logging.info(f"Split into {len(self.df_train)} train and {len(self.df_test)} test samples")