import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the style for all plots
sns.set_theme(style="whitegrid")

# Define global variables for input/output paths
INPUT_FILEPATH = os.path.join('..', 'aggregation_summary', 'aggregation_summary_report.csv')
OUTPUT_SUBDIR = os.path.join('..', 'aggregation_summary')

# Ensure output directory exists
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)

def calculate_f1_score(row):
    """Calculate F1 score from confusion matrix values."""
    precision = row['confusion_tp'] / (row['confusion_tp'] + row['confusion_fp']) if (row['confusion_tp'] + row['confusion_fp']) > 0 else 0
    recall = row['confusion_tp'] / (row['confusion_tp'] + row['confusion_fn']) if (row['confusion_tp'] + row['confusion_fn']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def load_and_preprocess_data():
    """Load and preprocess the CSV data."""
    df = pd.read_csv(INPUT_FILEPATH)
    
    # Calculate F1 score for each row
    df['f1_score'] = df.apply(calculate_f1_score, axis=1)
    
    # Create clean model names by removing common suffixes
    df['model_name_clean'] = df['model_name'].str.replace(':.*$', '', regex=True)
    
    return df

def plot_f1_by_model_prompt(df):
    """Plot F1 scores grouped by model and prompt type."""
    # Create pivot table for plotting
    plot_data = df.pivot(index='model_name_clean', columns='prompt_type', values='f1_score')
    
    # Sort models by their maximum F1 score across prompt types
    plot_data['max_f1'] = plot_data.max(axis=1)
    plot_data = plot_data.sort_values('max_f1', ascending=False)
    plot_data = plot_data.drop('max_f1', axis=1)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    plot_data.plot(kind='bar', width=0.8)
    
    plt.title('F1 Scores by Model and Prompt Type')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.legend(title='Prompt Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_SUBDIR, 'f1_by_model_prompt.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_by_model_prompt(df):
    """Plot accuracy scores grouped by model and prompt type."""
    # Create pivot table for plotting
    plot_data = df.pivot(index='model_name_clean', columns='prompt_type', values='prediction_accuracy')
    
    # Sort models by their maximum accuracy across prompt types
    plot_data['max_acc'] = plot_data.max(axis=1)
    plot_data = plot_data.sort_values('max_acc', ascending=False)
    plot_data = plot_data.drop('max_acc', axis=1)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    plot_data.plot(kind='bar', width=0.8)
    
    plt.title('Accuracy by Model and Prompt Type')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.legend(title='Prompt Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_SUBDIR, 'accuracy_by_model_prompt.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_bottom_f1_scores(df):
    """Plot top 10 and bottom 10 F1 scores."""
    # Create model+prompt combination identifier
    df['model_prompt'] = df['model_name_clean'] + ' + ' + df['prompt_type']
    
    # Get top 10 and bottom 10 combinations
    top_10 = df.nlargest(10, 'f1_score')
    bottom_10 = df.nsmallest(10, 'f1_score')
    plot_data = pd.concat([top_10, bottom_10])
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(plot_data)), plot_data['f1_score'])
    
    # Color the bars (blue for top 10, red for bottom 10)
    for i, bar in enumerate(bars):
        bar.set_color('blue' if i < 10 else 'red')
    
    plt.title('Top 10 and Bottom 10 F1 Scores by Model + Prompt Type')
    plt.xlabel('Model + Prompt Type')
    plt.ylabel('F1 Score')
    plt.xticks(range(len(plot_data)), plot_data['model_prompt'], rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_SUBDIR, 'top_bottom_f1_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_vs_duration(df):
    """Plot F1 scores vs API duration for top 10 and bottom 10 combinations."""
    # Create model+prompt combination identifier
    df['model_prompt'] = df['model_name_clean'] + ' + ' + df['prompt_type']
    
    # Get top 10 and bottom 10 combinations
    top_10 = df.nlargest(10, 'f1_score')
    bottom_10 = df.nsmallest(10, 'f1_score')
    plot_data = pd.concat([top_10, bottom_10])
    
    # Create scatter plot
    plt.figure(figsize=(15, 8))
    
    # Plot points
    plt.scatter(plot_data['api_total_duration_sec'], plot_data['f1_score'],
                c=['blue' if i < 10 else 'red' for i in range(20)],
                s=100)
    
    # Add labels for each point
    for idx, row in plot_data.iterrows():
        plt.annotate(row['model_prompt'], 
                    (row['api_total_duration_sec'], row['f1_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.title('F1 Score vs API Duration (Top 10 and Bottom 10)')
    plt.xlabel('API Total Duration (seconds)')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_SUBDIR, 'f1_vs_duration.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to execute all visualizations."""
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Generate all plots
    plot_f1_by_model_prompt(df)
    plot_accuracy_by_model_prompt(df)
    plot_top_bottom_f1_scores(df)
    plot_f1_vs_duration(df)

if __name__ == "__main__":
    main()