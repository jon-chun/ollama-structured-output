import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datetime import datetime
import argparse
import os
from typing import Tuple, Any
import logging

# Configure default paths for data and output
# We structure the paths relative to the project root for better organization
INPUT_DATA_FILENAME = 'vignettes_renamed_clean.csv'
INPUT_DATA_FILEPATH = os.path.join('data', INPUT_DATA_FILENAME)

# Create a dedicated subdirectory for SHAP reports to keep outputs organized
OUTPUT_REPORT_SUBDIR = 'feature_importance_shap'
OUTPUT_DATA_FILEPATH = os.path.join('data', OUTPUT_REPORT_SUBDIR)

# Set up logging with a format that includes timestamps for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argparse() -> argparse.Namespace:
    """
    Configure command-line argument parsing with sensible defaults.
    The default paths can be overridden via command line if needed.
    """
    parser = argparse.ArgumentParser(description="Generate SHAP feature importance report.")
    parser.add_argument(
        "--input_file", 
        default=INPUT_DATA_FILEPATH,
        help=f"Path to the input CSV file (default: {INPUT_DATA_FILEPATH})"
    )
    parser.add_argument(
        "--output_dir", 
        default=OUTPUT_DATA_FILEPATH,
        help=f"Directory to save reports (default: {OUTPUT_DATA_FILEPATH})"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2, 
        help="Test set size ratio"
    )
    return parser.parse_args()

def validate_paths(input_file: str, output_dir: str) -> None:
    """
    Ensure that input file exists and create output directory if needed.
    Now includes helpful error messages about the default paths.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Please ensure the file exists or specify a different path using --input_file"
        )
    
    # Create the output directory structure if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory confirmed/created at: {output_dir}")
    except Exception as e:
        raise Exception(f"Failed to create output directory {output_dir}: {str(e)}")

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the data, with enhanced logging about the default dataset.
    """
    logger.info(f"Loading dataset from: {file_path}")
    if file_path == INPUT_DATA_FILEPATH:
        logger.info("Using default dataset location")
    
    df = pd.read_csv(file_path)
    
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    logger.info(f"Dropped {rows_dropped} rows with NA values")
    
    if "target" not in df.columns:
        raise ValueError(
            "Target column not found in dataset. "
            "Please ensure the input file contains a 'target' column."
        )
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Handle categorical variables with improved logging
    categorical_columns = X.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col].astype(str))
        logger.info(f"Encoded categorical column: {col}")
    
    return X, y

def save_shap_results(shap_values: Any, X_test: pd.DataFrame, output_dir: str) -> None:
    """
    Save SHAP analysis results with organized file naming that includes the default directory.
    """
    # Create a timestamp-based subdirectory for this run
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, now)
    os.makedirs(run_dir, exist_ok=True)
    
    text_path = os.path.join(run_dir, "feature_importance_report.txt")
    png_path = os.path.join(run_dir, "feature_importance_plot.png")
    
    # Save summary plot with improved settings
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    
    # Enhanced text report
    with open(text_path, "w") as f:
        f.write("SHAP Feature Importance Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {now}\n")
        f.write(f"Input Data: {INPUT_DATA_FILEPATH}\n\n")
        f.write("Feature Importance Summary:\n")
        f.write("-" * 30 + "\n")
        
        # Calculate and store mean absolute SHAP values per feature
        mean_shap_values = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values.values).mean(0)
        }).sort_values('importance', ascending=False)
        
        f.write(mean_shap_values.to_string())
    
    logger.info(f"SHAP report saved in directory: {run_dir}")
    logger.info(f"Files generated:\n  - Text report: {text_path}\n  - Plot: {png_path}")

def main():
    """
    Main execution function with enhanced error handling and modified cross-validation approach.
    """
    try:
        args = setup_argparse()
        validate_paths(args.input_file, args.output_dir)
        
        X, y = load_and_preprocess_data(args.input_file)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed
        )
        
        # Train model with basic parameter tuning
        model = xgb.XGBClassifier(
            random_state=args.random_seed,
            n_estimators=100,
            learning_rate=0.1,
            use_label_encoder=False,  # Add this to prevent warning
            eval_metric='logloss'     # Add this to prevent warning
        )
        
        # Instead of using cross_val_score, we'll implement a manual k-fold validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=args.random_seed)
        cv_scores = []
        
        logger.info("Performing cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            # Split data for this fold
            fold_X_train = X_train.iloc[train_idx]
            fold_y_train = y_train.iloc[train_idx]
            fold_X_val = X_train.iloc[val_idx]
            fold_y_val = y_train.iloc[val_idx]
            
            # Train model on this fold
            fold_model = xgb.XGBClassifier(
                random_state=args.random_seed,
                n_estimators=100,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # Fit and evaluate
            fold_model.fit(fold_X_train, fold_y_train)
            fold_score = fold_model.score(fold_X_val, fold_y_val)
            cv_scores.append(fold_score)
            
            logger.info(f"Fold {fold} accuracy: {fold_score:.3f}")
        
        # Calculate and log cross-validation statistics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        logger.info(f"Cross-validation accuracy: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
        
        # Train final model on full training set
        logger.info("Training final model on full training set...")
        model.fit(X_train, y_train)
        
        # Generate predictions and classification report
        y_pred = model.predict(X_test)
        logger.info("\nModel Performance on Test Set:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Calculate SHAP values with error handling
        try:
            logger.info("Calculating SHAP values...")
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            save_shap_results(shap_values, X_test, args.output_dir)
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
if __name__ == "__main__":
    main()