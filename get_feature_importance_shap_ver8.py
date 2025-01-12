import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report
)

import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
INPUT_DATA_FILENAME = 'vignettes_renamed_clean.csv'
INPUT_DATA_FILEPATH = os.path.join('data', INPUT_DATA_FILENAME)

OUTPUT_REPORT_SUBDIR = 'feature_importance_shap'
OUTPUT_DATA_FILEPATH = os.path.join('data', OUTPUT_REPORT_SUBDIR)

COLS_IGNORE_LS = ['short_text_summary', 'long_text_summary']
COL_PREDICTION = ['target']

KFOLD_N_SPLITS = 5

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

# File handler (debug.log)
debug_file_handler = logging.FileHandler('debug.log', mode='w')
debug_file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
debug_file_handler.setFormatter(file_formatter)
logger.addHandler(debug_file_handler)

# -----------------------------------------------------------------------------
# ARGPARSE
# -----------------------------------------------------------------------------
def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP feature importance with manual transformation + XGBoost (no early stopping).")
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
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in XGBoost (default=100)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for XGBoost (default=0.1)"
    )
    return parser.parse_args()

# -----------------------------------------------------------------------------
# PATH VALIDATION
# -----------------------------------------------------------------------------
def validate_paths(input_file: str, output_dir: str) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory confirmed/created at: {output_dir}")
    except Exception as e:
        raise Exception(f"Failed to create output directory {output_dir}: {str(e)}")

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
def load_and_preprocess_data(file_path: str):
    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    
    # Drop ignored columns
    for col in COLS_IGNORE_LS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info(f"Dropped column: {col}")
    
    # Drop NaN rows
    init_rows = len(df)
    df.dropna(inplace=True)
    logger.info(f"Dropped {init_rows - len(df)} rows due to NA values.")
    
    # Ensure target column
    if COL_PREDICTION[0] not in df.columns:
        raise ValueError(f"Missing target column '{COL_PREDICTION[0]}'")
    
    X = df.drop(columns=COL_PREDICTION)
    y = df[COL_PREDICTION[0]]
    logger.info(f"Final dataset has {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y

# -----------------------------------------------------------------------------
# PREPROCESSOR PIPELINE (NO CLASSIFIER)
# -----------------------------------------------------------------------------
def create_preprocessor(numeric_features, categorical_features):
    """
    Create a pipeline that only transforms data:
    - StandardScaler for numeric
    - OneHotEncoder for categorical
    """
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

# -----------------------------------------------------------------------------
# NEW: generate_report() FUNCTION
# -----------------------------------------------------------------------------
def generate_report(
    run_dir: str,
    training_results: Dict[str, Any],
    shap_values_array: np.ndarray,
    feature_names: list
):
    """
    Writes a text file 'report.txt' in run_dir, summarizing:
      - Cross-validation metrics
      - Test performance
      - Confusion matrix
      - Feature Importance Ranking (descending mean(|SHAP|))
    """
    # Create a DataFrame of mean(|SHAP|)
    mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    importance_df.reset_index(drop=True, inplace=True)
    
    text_path = os.path.join(run_dir, "report.txt")
    with open(text_path, "w") as f:
        # Write Cross-Validation Metrics
        f.write("Cross-Validation Metrics\n")
        f.write("========================\n")
        cv_scores = training_results.get('cv_scores', [])
        for fold_idx, scores_dict in enumerate(cv_scores, 1):
            f.write(
                f"Fold {fold_idx} => ACC: {scores_dict['acc']:.3f}, "
                f"RECALL: {scores_dict['recall']:.3f}, F1: {scores_dict['f1']:.3f}, "
                f"AUC: {scores_dict['auc']:.3f}\n"
            )
        
        # Write Test Set Performance
        f.write("\nTest Set Performance\n")
        f.write("====================\n")
        classification_rep = training_results.get('classification_report', '')
        f.write(classification_rep)
        
        conf_matrix = training_results.get('test_confusion_matrix', None)
        f.write(f"\nConfusion Matrix:\n{conf_matrix}\n")
        
        # Write Feature Importance Ranking
        f.write("\nFeature Importance Ranking (Descending SHAP)\n")
        f.write("=============================================\n")
        for rank, row in importance_df.iterrows():
            f.write(f"#{rank+1:2d} -- {row['mean_abs_shap']:.5f} -- {row['feature']:30s}\n")
    
    logger.info(f"Text report generated at: {text_path}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    try:
        args = setup_argparse()
        validate_paths(args.input_file, args.output_dir)
        
        # Load data
        X, y = load_and_preprocess_data(args.input_file)
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create preprocessor pipeline
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=args.random_seed
        )
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS, shuffle=True, random_state=args.random_seed)
        
        logger.info(f"Starting {KFOLD_N_SPLITS}-fold cross-validation with MANUAL transformations + XGB (no early stopping)")
        
        training_results = {'cv_scores': []}
        
        # Cross-validation loop
        fold_count = 0
        for train_idx, val_idx in skf.split(X_train, y_train):
            fold_count += 1
            logger.info(f"===== Fold {fold_count}/{KFOLD_N_SPLITS} =====")
            
            # Create fold data
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Fit the preprocessor on X_fold_train only, transform train and val
            preprocessor.fit(X_fold_train)
            X_fold_train_transformed = preprocessor.transform(X_fold_train)
            X_fold_val_transformed = preprocessor.transform(X_fold_val)
            
            # Create XGBoost model for this fold
            xgb_model = XGBClassifier(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                random_state=args.random_seed,
                eval_metric="logloss"  # no early stopping
            )
            
            # Fit (no early_stopping_rounds)
            xgb_model.fit(
                X_fold_train_transformed,
                y_fold_train,
                eval_set=[(X_fold_val_transformed, y_fold_val)],
                verbose=False
            )
            
            # Predict on val
            y_val_pred = xgb_model.predict(X_fold_val_transformed)
            acc = accuracy_score(y_fold_val, y_val_pred)
            rec = recall_score(y_fold_val, y_val_pred, zero_division=0)
            f1_ = f1_score(y_fold_val, y_val_pred, zero_division=0)
            try:
                auc_ = roc_auc_score(y_fold_val, xgb_model.predict_proba(X_fold_val_transformed)[:,1])
            except:
                auc_ = np.nan
            
            logger.info(f"Fold {fold_count} => ACC: {acc:.3f}, REC: {rec:.3f}, F1: {f1_:.3f}, AUC: {auc_:.3f}")
            training_results['cv_scores'].append({'acc': acc,'recall': rec,'f1': f1_,'auc': auc_})
        
        # Train final model on ALL X_train
        logger.info("Fitting final model on full training set (no early stopping).")
        
        # Fit preprocessor on full training set
        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        final_model = XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=args.random_seed,
            eval_metric="logloss"
        )
        
        final_model.fit(
            X_train_transformed,
            y_train,
            eval_set=[(X_test_transformed, y_test)],
            verbose=False
        )
        
        # Evaluate on test
        y_test_pred = final_model.predict(X_test_transformed)
        test_report = classification_report(y_test, y_test_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_test_pred)
        
        logger.info("===== Test Set Performance =====")
        logger.info("\n" + test_report)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        training_results['classification_report'] = test_report
        training_results['test_confusion_matrix'] = cm
        
        # Retrieve feature names from the fitted preprocessor
        feature_names = preprocessor.get_feature_names_out()
        
        # Prepare SHAP
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer(X_test_transformed)
        
        # Convert Explanation -> array for plotting
        shap_values_array = shap_values.values
        
        # Create run_dir for outputs
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # 1) SHAP summary plot
        shap_summary_path = os.path.join(run_dir, "shap_summary.png")
        plt.figure(figsize=(12, 8), dpi=150)
        shap.summary_plot(
            shap_values_array, 
            X_test_transformed, 
            feature_names=feature_names, 
            show=False
        )
        plt.tight_layout()
        plt.savefig(shap_summary_path, bbox_inches='tight')
        plt.close()
        
        # 2) Dependence plot (top feature)
        mean_abs = np.mean(np.abs(shap_values_array), axis=0)
        top_idx = int(np.argmax(mean_abs))
        
        shap_depend_path = os.path.join(run_dir, f"shap_dependence_top_feature.png")
        plt.figure(figsize=(8, 6), dpi=150)
        shap.dependence_plot(
            top_idx,
            shap_values_array,
            X_test_transformed,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(shap_depend_path, bbox_inches='tight')
        plt.close()
        
        # 3) Save raw SHAP values
        shap_df = pd.DataFrame(shap_values_array, columns=feature_names)
        shap_csv_path = os.path.join(run_dir, "shap_values.csv")
        shap_df.to_csv(shap_csv_path, index=False)
        
        # 4) Generate the text report (factor out)
        generate_report(
            run_dir=run_dir,
            training_results=training_results,
            shap_values_array=shap_values_array,
            feature_names=feature_names
        )
        
        logger.info(f"SHAP analysis and results saved in: {run_dir}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
