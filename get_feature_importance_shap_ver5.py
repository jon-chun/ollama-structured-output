import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

# Sklearn and XGBoost
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report
)

# SHAP and Visualization
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

# Single reference for KFold splits
KFOLD_N_SPLITS = 5

# -----------------------------------------------------------------------------
# LOGGING CONFIG
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels; fine-tune with handlers

# Create console handler with a higher log level (INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

# Create debug file handler
debug_file_handler = logging.FileHandler('debug.log', mode='w')
debug_file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
debug_file_handler.setFormatter(file_formatter)
logger.addHandler(debug_file_handler)

# -----------------------------------------------------------------------------
# ARGPARSE
# -----------------------------------------------------------------------------
def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP feature importance report with an XGBoost Pipeline.")
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
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=10,
        help="Early stopping rounds for XGBoost (default=10)"
    )
    return parser.parse_args()

# -----------------------------------------------------------------------------
# PATH VALIDATION
# -----------------------------------------------------------------------------
def validate_paths(input_file: str, output_dir: str) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Please ensure the file exists or specify a different path using --input_file"
        )
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory confirmed/created at: {output_dir}")
    except Exception as e:
        raise Exception(f"Failed to create output directory {output_dir}: {str(e)}")

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
def load_and_preprocess_data(file_path: str):
    """
    Load the CSV, drop ignored columns, and separate X and y.
    We do not do label encoding or scaling here. Instead, this
    is done in the pipeline.
    """
    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    
    # Drop columns we want to ignore
    initial_cols = df.columns
    for col in COLS_IGNORE_LS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info(f"Dropped column: {col}")
        else:
            logger.debug(f"Column to ignore not in dataset: {col}")
    
    # Drop NaNs
    initial_rows = len(df)
    df.dropna(inplace=True)
    logger.info(f"Dropped {initial_rows - len(df)} rows due to NA values.")
    
    # Ensure target column exists
    if COL_PREDICTION[0] not in df.columns:
        raise ValueError(f"Target column '{COL_PREDICTION[0]}' not found. Available: {list(df.columns)}")
    
    X = df.drop(columns=COL_PREDICTION)
    y = df[COL_PREDICTION[0]]
    logger.info(f"Final dataset has {X.shape[0]} rows and {X.shape[1]} features.")
    
    return X, y

# -----------------------------------------------------------------------------
# PIPELINE CREATION
# -----------------------------------------------------------------------------
def create_pipeline(n_estimators: int, learning_rate: float, random_seed: int):
    """
    Create a pipeline with:
    1) ColumnTransformer for numeric vs. categorical columns
    2) XGBClassifier with given hyperparameters
    """
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # We won't know columns until after reading data, so we build
    # the ColumnTransformer later (in main) after we see X's dtypes.
    
    # Just define the model for the pipeline here
    xgb_clf = XGBClassifier(
        random_state=random_seed,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        # We'll pass early_stopping params during .fit(...) calls
        use_label_encoder=False,  # often recommended since XGB 1.0+
        eval_metric='logloss'
    )
    
    pipeline = Pipeline([
        # The 'preprocessor' step will be inserted in main() once we know columns
        ('preprocessor', None),  
        ('classifier', xgb_clf)
    ])
    
    return pipeline

# -----------------------------------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------------------------------
def save_shap_results(
    explainer: shap.Explainer,
    shap_values: shap._explanation.Explanation,
    X_test: pd.DataFrame, 
    output_dir: str,
    training_results: Dict[str, Any],
    pipeline_feature_names: list,
    run_id: str
):
    """
    Save SHAP plots, text summary, and raw SHAP values.
    Also demonstrates PDP / ICE plots.
    """
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # ---- 1) Text-based summary report ----
    text_path = os.path.join(run_dir, "feature_importance_report.txt")
    with open(text_path, "w") as f:
        f.write("SHAP Feature Importance Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Cross-validation results:\n")
        for fold_idx, scores_dict in enumerate(training_results['cv_scores'], 1):
            f.write(f"Fold {fold_idx} => "
                    f"ACC: {scores_dict['acc']:.3f}, "
                    f"RECALL: {scores_dict['recall']:.3f}, "
                    f"F1: {scores_dict['f1']:.3f}, "
                    f"AUC: {scores_dict['auc']:.3f}\n")
        
        f.write("\nConfusion Matrix (Test Set):\n")
        f.write(str(training_results['test_confusion_matrix']))
        f.write("\n\n")
        
        f.write("Test Set Classification Report:\n")
        f.write(training_results['classification_report'])
    
    # ---- 2) SHAP Summary Plot ----
    png_path = os.path.join(run_dir, "shap_summary_plot.png")
    plt.figure(figsize=(12, 8), dpi=200)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP summary plot saved: {png_path}")

    # ---- 3) SHAP Dependence Plot for top feature (example) ----
    # Identify top feature by mean(|SHAP|)
    mean_shap_abs = np.mean(np.abs(shap_values.values), axis=0)
    top_feature_idx = np.argmax(mean_shap_abs)
    
    # Safety check in case we can't find a valid top feature
    if 0 <= top_feature_idx < len(pipeline_feature_names):
        top_feature = pipeline_feature_names[top_feature_idx]
        shap_dep_path = os.path.join(run_dir, f"shap_dependence_{top_feature}.png")
        
        # Attempt to plot
        plt.figure(figsize=(10, 6), dpi=200)
        shap.dependence_plot(
            top_feature_idx, 
            shap_values, 
            X_test, 
            feature_names=pipeline_feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(shap_dep_path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP dependence plot for top feature '{top_feature}' saved: {shap_dep_path}")
    
    # ---- 4) Save raw SHAP values to CSV ----
    shap_csv_path = os.path.join(run_dir, "shap_values.csv")
    shap_df = pd.DataFrame(shap_values.values, columns=pipeline_feature_names)
    shap_df.to_csv(shap_csv_path, index=False)
    logger.info(f"Raw SHAP values saved: {shap_csv_path}")
    
    # ---- 5) Partial Dependence and ICE Plots (example) ----
    # For demonstration, pick a single feature from the top or pick one by name
    pdp_feature = pipeline_feature_names[min(top_feature_idx, len(pipeline_feature_names) - 1)]
    pdp_fig_path = os.path.join(run_dir, f"pdp_ice_{pdp_feature}.png")
    
    # For PDP, we need the trained pipeline (with preprocessor). We'll assume
    # X_test is the raw features (pre-transform). We'll pass the pipeline as the estimator.
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    try:
        PartialDependenceDisplay.from_estimator(
            estimator=explainer.model,  # or the pipeline if referencing the final object
            X=X_test,
            features=[pdp_feature],
            kind="both",  # "both" => PDP + ICE
            ax=ax
        )
        plt.title(f"PDP / ICE for '{pdp_feature}'")
        plt.savefig(pdp_fig_path, bbox_inches='tight')
        plt.close()
        logger.info(f"PDP/ICE plot for '{pdp_feature}' saved: {pdp_fig_path}")
    except Exception as e:
        logger.debug(f"Could not generate PDP/ICE plot for feature: {pdp_feature}. Reason: {e}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    try:
        args = setup_argparse()
        
        # 1. Validate paths
        validate_paths(args.input_file, args.output_dir)
        
        # 2. Load data
        X, y = load_and_preprocess_data(args.input_file)
        
        # 3. Identify numeric & categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns.to_list()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.to_list()
        
        # 4. Create pipeline
        pipeline = create_pipeline(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            random_seed=args.random_seed
        )
        
        # Insert the ColumnTransformer now that we know numeric/cat columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        pipeline.steps[0] = ('preprocessor', preprocessor)
        
        # 5. Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, 
            random_state=args.random_seed, 
            stratify=y  # Good practice for classification
        )
        
        # 6. K-Fold (Stratified) on the Training Split
        skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS, shuffle=True, random_state=args.random_seed)
        logger.info(f"Starting {KFOLD_N_SPLITS}-fold cross-validation with early stopping...")
        
        training_results = {
            'cv_scores': [],  # each fold will have multiple metrics
            'classification_report': '',
            'test_confusion_matrix': None
        }
        
        fold_count = 0
        for train_idx, val_idx in skf.split(X_train, y_train):
            fold_count += 1
            logger.info(f"===== Fold {fold_count}/{KFOLD_N_SPLITS} =====")
            
            # Split
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Fit pipeline with early stopping
            pipeline.fit(
                X_fold_train, y_fold_train,
                classifier__eval_set=[(X_fold_val, y_fold_val)],
                classifier__early_stopping_rounds=args.early_stopping_rounds,
                classifier__verbose=False  # set to True for debugging
            )
            
            # Predict on validation fold
            y_val_pred = pipeline.predict(X_fold_val)
            acc = accuracy_score(y_fold_val, y_val_pred)
            rec = recall_score(y_fold_val, y_val_pred, zero_division=0)
            f1 = f1_score(y_fold_val, y_val_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_fold_val, pipeline.predict_proba(X_fold_val)[:,1])
            except:
                auc = np.nan
            
            logger.info(f"Fold {fold_count} => ACC: {acc:.3f}, REC: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
            
            training_results['cv_scores'].append({
                'acc': acc,
                'recall': rec,
                'f1': f1,
                'auc': auc
            })
        
        # 7. Train final model on full training set
        logger.info("Retraining pipeline on the entire training set (with early stopping)...")
        pipeline.fit(
            X_train, y_train,
            classifier__eval_set=[(X_test, y_test)],
            classifier__early_stopping_rounds=args.early_stopping_rounds,
            classifier__verbose=False
        )
        
        # 8. Evaluate on test set
        y_test_pred = pipeline.predict(X_test)
        test_report = classification_report(y_test, y_test_pred, zero_division=0)
        training_results['classification_report'] = test_report
        cm = confusion_matrix(y_test, y_test_pred)
        training_results['test_confusion_matrix'] = cm
        
        logger.info("===== Test Set Performance =====")
        logger.info("\n" + test_report)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # 9. SHAP Analysis
        logger.info("Calculating SHAP values on test set...")
        
        # We can extract the final XGBoost model from the pipeline
        xgb_model = pipeline['classifier']
        
        # But to get correct SHAP input, we must transform X_test
        # because the pipeline includes transformations.
        X_test_transformed = pipeline['preprocessor'].transform(X_test)
        
        # Create TreeExplainer
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_test_transformed)  
        
        # The pipeline's preprocessor changes feature names
        # Let's retrieve them for correct labeling
        feature_names = pipeline['preprocessor'].get_feature_names_out()
        
        # 10. Save results
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_shap_results(
            explainer,
            shap_values,
            X_test,                 # untransformed data for shap.summary_plot usage
            args.output_dir,
            training_results,
            feature_names.tolist(), # shap needs array-like feature names
            run_id
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
