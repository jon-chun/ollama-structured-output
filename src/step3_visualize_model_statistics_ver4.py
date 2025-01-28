import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the style and color palette
sns.set_theme(style="whitegrid")
COLORS = ['#4c72b0', '#dd8452', '#55a868']  # seaborn "deep" palette subset

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
    
    # Calculate total tokens
    df['total_tokens'] = df['api_prompt_eval_count'] + df['api_eval_count']
    
    # Group by model and prompt type to handle duplicates
    df_grouped = df.groupby(['model_name_clean', 'prompt_type']).agg({
        'f1_score': 'mean',
        'accuracy': 'mean',
        'api_total_duration_sec': 'mean',
        'total_tokens': 'mean'
    }).reset_index()
    
    return df_grouped

def plot_f1_by_model_prompt(df):
    """Plot F1 scores grouped by model and prompt type."""
    # Get top 10 models based on maximum F1 score
    max_f1_by_model = df.groupby('model_name_clean')['f1_score'].max()
    top_10_models = max_f1_by_model.nlargest(10).index
    bottom_10_models = max_f1_by_model.nsmallest(10).index
    selected_models = pd.concat([pd.Series(top_10_models), pd.Series(bottom_10_models)])
    
    # Filter data for selected models
    df_filtered = df[df['model_name_clean'].isin(selected_models)]
    
    # Create pivot table for plotting
    plot_data = df_filtered.pivot(index='model_name_clean', columns='prompt_type', values='f1_score')
    
    # Sort models by their maximum F1 score
    plot_data['max_f1'] = plot_data.max(axis=1)
    plot_data = plot_data.sort_values('max_f1', ascending=False)
    plot_data = plot_data.drop('max_f1', axis=1)
    
    # Create the plot
    plt.figure(figsize=(15, 6))  # Reduced height
    plot_data.plot(kind='bar', width=0.8, color=COLORS)
    
    plt.title('F1 Scores by Model and Prompt Type (Top 10 and Bottom 10)')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.legend(title='Prompt Type', bbox_to_anchor=(0, 0), loc='lower left', fontsize=7.5)  # 75% of original 10pt font
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, plot_data.values.max() * 1.1)  # Adjust y-axis limit
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_SUBDIR, 'f1_by_model_prompt.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_by_model_prompt(df):
    """Plot accuracy scores grouped by model and prompt type."""
    # Get top 10 models based on maximum accuracy
    max_acc_by_model = df.groupby('model_name_clean')['accuracy'].max()
    top_10_models = max_acc_by_model.nlargest(10).index
    bottom_10_models = max_acc_by_model.nsmallest(10).index
    selected_models = pd.concat([pd.Series(top_10_models), pd.Series(bottom_10_models)])
    
    # Filter data for selected models
    df_filtered = df[df['model_name_clean'].isin(selected_models)]
    
    # Create pivot table for plotting
    plot_data = df_filtered.pivot(index='model_name_clean', columns='prompt_type', values='accuracy')
    
    # Sort models by their maximum accuracy
    plot_data['max_acc'] = plot_data.max(axis=1)
    plot_data = plot_data.sort_values('max_acc', ascending=False)
    plot_data = plot_data.drop('max_acc', axis=1)
    
    # Create the plot
    plt.figure(figsize=(15, 6))  # Reduced height
    plot_data.plot(kind='bar', width=0.8, color=COLORS)
    
    plt.title('Accuracy by Model and Prompt Type (Top 10 and Bottom 10)')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.legend(title='Prompt Type', bbox_to_anchor=(0, 0), loc='lower left', fontsize=7.5)  # 75% of original 10pt font
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, plot_data.values.max() * 1.1)  # Adjust y-axis limit
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
    bottom_10 = df.nsmallest(10, 'f1_score').sort_values('f1_score', ascending=False)  # Sort descending
    plot_data = pd.concat([top_10, bottom_10])
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Create bar plot with different colors for top and bottom
    colors = ['#4c72b0'] * 10 + ['#dd8452'] * 10  # Use consistent colors
    bars = plt.bar(range(len(plot_data)), plot_data['f1_score'], color=colors)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Add vertical line between top 10 and bottom 10
    plt.axvline(x=9.5, color='red', alpha=0.7, linestyle='--')
    
    # Add "Top 10" and "Bottom 10" labels
    max_height = plot_data['f1_score'].max()
    plt.text(4.5, max_height * 0.9, 'TOP 10',
             ha='center', va='center', fontsize=16, fontweight='bold', alpha=0.3)
    plt.text(14.5, max_height * 0.9, 'BOTTOM 10',
             ha='center', va='center', fontsize=16, fontweight='bold', alpha=0.3)
    
    plt.title('Top 10 and Bottom 10 F1 Scores by Model + Prompt Type')
    plt.xlabel('Model + Prompt Type')
    plt.ylabel('F1 Score')
    plt.xticks(range(len(plot_data)), plot_data['model_prompt'], rotation=45, ha='right')
    plt.legend(['Top 10', 'Bottom 10'], bbox_to_anchor=(0, 0), loc='lower left',
              fontsize=7.5, facecolor='white', edgecolor='black')  # Added white background
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_SUBDIR, 'top_bottom_f1_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_vs_duration(df):
    """Plot F1 scores vs API duration for all combinations with different markers per prompt type."""
    # Create scatter plot with larger size
    plt.figure(figsize=(15, 8))
    
    # Increase font size
    plt.rcParams.update({'font.size': 14})
    
    # Define markers for each prompt type
    markers = {'standard': 'o', 'zero-shot': 's', 'few-shot': '^'}
    
    # Plot all points by prompt type
    for prompt_type, marker in markers.items():
        mask = df['prompt_type'] == prompt_type
        subset = df[mask]
        if not subset.empty:  # Only plot if we have data
            plt.scatter(subset['api_total_duration_sec'], 
                       subset['f1_score'],
                       marker=marker, s=200, label=prompt_type)
    
    # Add labels for each point with background
    for idx, row in df.iterrows():
        plt.annotate(row['model_name_clean'],
                    (row['api_total_duration_sec'], row['f1_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title('F1 Score vs API Duration (All Models)', fontsize=16)
    plt.xlabel('API Total Duration (seconds)', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.legend(fontsize=12, bbox_to_anchor=(0, 0), loc='lower left',
              facecolor='white', edgecolor='black')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_SUBDIR, 'f1_vs_duration.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset font size
    plt.rcParams.update({'font.size': 10})

def plot_f1_vs_tokens(df):
    """Plot F1 scores vs total tokens for all combinations with different markers per prompt type."""
    # Create scatter plot with larger size
    plt.figure(figsize=(15, 8))
    
    # Increase font size
    plt.rcParams.update({'font.size': 14})
    
    # Define markers for each prompt type
    markers = {'standard': 'o', 'zero-shot': 's', 'few-shot': '^'}
    
    # Plot all points by prompt type
    for prompt_type, marker in markers.items():
        mask = df['prompt_type'] == prompt_type
        subset = df[mask]
        if not subset.empty:  # Only plot if we have data
            plt.scatter(subset['total_tokens'], 
                       subset['f1_score'],
                       marker=marker, s=200, label=prompt_type)
    
    # Add labels for each point with background
    for idx, row in df.iterrows():
        plt.annotate(row['model_name_clean'],
                    (row['total_tokens'], row['f1_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title('F1 Score vs Total Tokens (All Models)', fontsize=16)
    plt.xlabel('Total Tokens', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.legend(fontsize=12, bbox_to_anchor=(0, 0), loc='lower left',
              facecolor='white', edgecolor='black')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_SUBDIR, 'f1_vs_tokens.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset font size
    plt.rcParams.update({'font.size': 10})

def main():
    """Main function to execute all visualizations."""
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Generate all plots
    plot_f1_by_model_prompt(df)
    plot_accuracy_by_model_prompt(df)
    plot_top_bottom_f1_scores(df)
    plot_f1_vs_duration(df)
    plot_f1_vs_tokens(df)

if __name__ == "__main__":
    main()