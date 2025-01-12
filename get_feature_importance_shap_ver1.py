import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import argparse
import os

# CLI Setup
parser = argparse.ArgumentParser(description="Generate SHAP feature importance report.")
parser.add_argument("input_file", help="Path to the input CSV file")
parser.add_argument("output_dir", help="Directory to save reports")
args = parser.parse_args()

# Utility function to save SHAP results
def save_shap_results(shap_values, X_test, output_dir):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = os.path.join(output_dir, f"report_feature_importance_shap_{now}.txt")
    png_path = os.path.join(output_dir, f"report_feature_importance_shap_{now}.png")

    # Save summary plot as PNG
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(png_path)
    
    # Save SHAP text summary
    with open(text_path, "w") as f:
        f.write("SHAP Feature Importance (Summary):\n")
        f.write(str(shap_values))
    
    print(f"SHAP report saved as:\n  Text: {text_path}\n  Plot: {png_path}")

# Main program
def main():
    # Load the dataset
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Preprocess data
    df = df.dropna()
    X = df.drop(columns=["target"])
    y = df["target"]
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Generate SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Save results
    save_shap_results(shap_values, X_test, args.output_dir)

if __name__ == "__main__":
    main()
