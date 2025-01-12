import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from datetime import datetime
import argparse
import os
from typing import Tuple, Any, Dict
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
    Enhanced data preprocessing to ensure all features are numeric.
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
    
    # Enhanced feature preprocessing
    feature_encodings = {}  # Store encoding information for reporting
    
    # Handle categorical variables with improved tracking
    categorical_columns = X.select_dtypes(include=["object"]).columns
    encoders = {}
    
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col].astype(str))
        # Store unique values mapping for reporting
        feature_encodings[col] = dict(zip(
            encoders[col].classes_,
            encoders[col].transform(encoders[col].classes_)
        ))
        logger.info(f"Encoded categorical column: {col}")
    
    # Scale numeric features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    return X, y, feature_encodings

def save_shap_results(
    shap_values: Any, 
    X_test: pd.DataFrame, 
    output_dir: str,
    training_results: Dict,
    feature_encodings: Dict
) -> None:
    """
    Enhanced results saving with training progression and feature encoding information.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, now)
    os.makedirs(run_dir, exist_ok=True)
    
    text_path = os.path.join(run_dir, "feature_importance_report.txt")
    png_path = os.path.join(run_dir, "feature_importance_plot.png")
    
    # Save summary plot with improved settings
    plt.figure(figsize=(12, 8), dpi=300)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    
    # Enhanced text report
    with open(text_path, "w") as f:
        f.write("SHAP Feature Importance Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {now}\n")
        f.write(f"Input Data: {INPUT_DATA_FILEPATH}\n\n")
        
        # Write training progression
        f.write("Training Progress\n")
        f.write("-" * 30 + "\n")
        f.write(f"Cross-validation results:\n")
        for fold, acc in enumerate(training_results['cv_scores'], 1):
            f.write(f"Fold {fold}: {acc:.3f}\n")
        f.write(f"\nMean CV accuracy: {training_results['cv_mean']:.3f} "
                f"(±{training_results['cv_std'] * 2:.3f})\n\n")
        
        # Write test set performance
        f.write("Test Set Performance\n")
        f.write("-" * 30 + "\n")
        f.write(training_results['classification_report'])
        f.write("\n\n")
        
        # Feature importance summary
        f.write("Feature Importance Summary\n")
        f.write("-" * 30 + "\n")
        mean_shap_values = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values.values).mean(0)
        }).sort_values('importance', ascending=False)
        f.write(mean_shap_values.to_string())
        
        # Feature encoding information
        f.write("\n\nFeature Encoding Information\n")
        f.write("-" * 30 + "\n")
        for feature, encoding in feature_encodings.items():
            f.write(f"\n{feature}:\n")
            for original, encoded in encoding.items():
                f.write(f"  {original} → {encoded}\n")
    
    logger.info(f"SHAP report saved in directory: {run_dir}")
    logger.info(f"Files generated:\n  - Text report: {text_path}\n  - Plot: {png_path}")




def main():
    """
    Enhanced main execution with progress tracking and comprehensive error handling.
    """
    try:
        args = setup_argparse()
        validate_paths(args.input_file, args.output_dir)
        
        # Load and preprocess data
        X, y, feature_encodings = load_and_preprocess_data(args.input_file)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed
        )
        
        # Store training results
        training_results = {
            'cv_scores': [],
            'cv_mean': 0,
            'cv_std': 0,
            'classification_report': ''
        }
        
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=args.random_seed)
        total_folds = 5
        
        logger.info("Starting cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            print(f"\rProgress: {fold}/{total_folds} folds completed... ({fold/total_folds*100:.1f}%)", end="")
            
            # Split data for this fold
            fold_X_train = X_train.iloc[train_idx]
            fold_y_train = y_train.iloc[train_idx]
            fold_X_val = X_train.iloc[val_idx]
            fold_y_val = y_train.iloc[val_idx]
            
            # Train model
            fold_model = xgb.XGBClassifier(
                random_state=args.random_seed,
                n_estimators=100,
                learning_rate=0.1,
                eval_metric='logloss'
            )
            
            fold_model.fit(fold_X_train, fold_y_train)
            fold_score = fold_model.score(fold_X_val, fold_y_val)
            training_results['cv_scores'].append(fold_score)
            
            logger.info(f"Fold {fold} accuracy: {fold_score:.3f}")
        
        print("\rCross-validation completed!                    ")
        
        # Calculate CV statistics
        training_results['cv_mean'] = np.mean(training_results['cv_scores'])
        training_results['cv_std'] = np.std(training_results['cv_scores'])
        logger.info(f"Cross-validation accuracy: {training_results['cv_mean']:.3f} "
                   f"(±{training_results['cv_std'] * 2:.3f})")
        
        # Train final model
        logger.info("Training final model on full training set...")
        print("\rTraining final model... (0%)", end="")
        
        model = xgb.XGBClassifier(
            random_state=args.random_seed,
            n_estimators=100,
            learning_rate=0.1,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        print("\rTraining final model... (100%)")
        
        # Generate predictions and classification report
        y_pred = model.predict(X_test)
        training_results['classification_report'] = classification_report(y_test, y_pred)
        logger.info("\nModel Performance on Test Set:")
        logger.info("\n" + training_results['classification_report'])
        
        # Calculate SHAP values
        try:
            logger.info("Calculating SHAP values...")
            print("\rCalculating SHAP values... (0%)", end="")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test)
            
            print("\rCalculating SHAP values... (100%)")
            
            save_shap_results(
                shap_values, 
                X_test, 
                args.output_dir, 
                training_results,
                feature_encodings
            )
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()