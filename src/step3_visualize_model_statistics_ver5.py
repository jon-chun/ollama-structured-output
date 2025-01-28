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
    try:
        df = pd.read_csv(INPUT_FILEPATH)
        
        # Verify required columns exist
        required_columns = ['model_name', 'confusion_tp', 'confusion_fp', 'confusion_fn']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Calculate F1 score for each row
        df['f1_score'] = df.apply(calculate_f1_score, axis=1)
        
        # Create clean model names by removing version and configuration info
        df['model_name_clean'] = df['model_name'].str.replace(':.*$', '', regex=True)
        
        # Calculate total tokens
        df['total_tokens'] = df['api_prompt_eval_count'] + df['api_eval_count']
        
        # Create prompt type if it doesn't exist (for testing)
        if 'prompt_type' not in df.columns:
            df['prompt_type'] = 'standard'  # default value
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find input file at {INPUT_FILEPATH}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The input CSV file is empty")
        raise
    except ValueError as ve:
        print(f"Error processing data: {ve}")
        raise
    except Exception as e:
        print(f"Unexpected error processing data: {e}")
        raise

def create_plot_base(figsize=(15, 8), fontsize=14):
    """Create base plot with common settings."""
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    return plt

def save_plot(filename, dpi=300):
    """Save plot with common settings."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIR, filename), dpi=dpi, bbox_inches='tight')
    plt.close()
    plt.rcParams.update({'font.size': 10})  # Reset font size

def plot_f1_by_model_prompt(df):
    """Plot F1 scores grouped by model and prompt type."""
    # Get top 10 and bottom 10 models based on maximum F1 score
    max_f1_by_model = df.groupby('model_name_clean')['f1_score'].max()
    
    # Convert index objects to Series before concatenation
    top_models = pd.Series(max_f1_by_model.nlargest(10).index)
    bottom_models = pd.Series(max_f1_by_model.nsmallest(10).index)
    selected_models = pd.concat([top_models, bottom_models])
    
    # Filter and prepare plot data
    df_filtered = df[df['model_name_clean'].isin(selected_models)]
    plot_data = df_filtered.pivot(index='model_name_clean', columns='prompt_type', values='f1_score')
    
    # Sort by maximum F1 score
    plot_data = plot_data.assign(max_f1=plot_data.max(axis=1)).sort_values('max_f1', ascending=False).drop('max_f1', axis=1)
    
    # Create and save plot
    create_plot_base(figsize=(15, 6))
    plot_data.plot(kind='bar', width=0.8, color=COLORS)
    
    plt.title('F1 Scores by Model and Prompt Type (Top 10 and Bottom 10)')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.legend(title='Prompt Type', bbox_to_anchor=(0, 0), loc='lower left', fontsize=7.5)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, plot_data.values.max() * 1.1)
    
    save_plot('f1_by_model_prompt.png')

def plot_accuracy_by_model_prompt(df):
    """Plot accuracy scores grouped by model and prompt type."""
    # Get top 10 and bottom 10 models based on maximum accuracy
    max_acc_by_model = df.groupby('model_name_clean')['accuracy'].max()
    
    # Convert index objects to Series before concatenation
    top_models = pd.Series(max_acc_by_model.nlargest(10).index)
    bottom_models = pd.Series(max_acc_by_model.nsmallest(10).index)
    selected_models = pd.concat([top_models, bottom_models])
    
    # Filter and prepare plot data
    df_filtered = df[df['model_name_clean'].isin(selected_models)]
    plot_data = df_filtered.pivot(index='model_name_clean', columns='prompt_type', values='accuracy')
    
    # Sort by maximum accuracy
    plot_data = plot_data.assign(max_acc=plot_data.max(axis=1)).sort_values('max_acc', ascending=False).drop('max_acc', axis=1)
    
    # Create and save plot
    create_plot_base(figsize=(15, 6))
    plot_data.plot(kind='bar', width=0.8, color=COLORS)
    
    plt.title('Accuracy by Model and Prompt Type (Top 10 and Bottom 10)')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.legend(title='Prompt Type', bbox_to_anchor=(0, 0), loc='lower left', fontsize=7.5)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, plot_data.values.max() * 1.1)
    
    save_plot('accuracy_by_model_prompt.png')

def plot_top_bottom_f1_scores(df):
    """Plot top 10 and bottom 10 F1 scores."""
    # Create model+prompt combination identifier and get top/bottom combinations
    df['model_prompt'] = df['model_name_clean'] + ' + ' + df['prompt_type']
    plot_data = pd.concat([
        df.nlargest(10, 'f1_score'),
        df.nsmallest(10, 'f1_score').sort_values('f1_score', ascending=False)
    ])
    
    # Create and customize plot
    create_plot_base()
    colors = ['#4c72b0'] * 10 + ['#dd8452'] * 10
    bars = plt.bar(range(len(plot_data)), plot_data['f1_score'], color=colors)
    
    # Add value labels and divider
    max_height = plot_data['f1_score'].max()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.axvline(x=9.5, color='red', alpha=0.7, linestyle='--')
    
    # Add section labels
    plt.text(4.5, max_height * 0.1, 'TOP 10',
             ha='center', va='center', fontsize=32, fontweight='bold', alpha=0.5)
    plt.text(14.5, max_height * 0.9, 'BOTTOM 10',
             ha='center', va='center', fontsize=32, fontweight='bold', alpha=0.5)
    
    plt.title('Top 10 and Bottom 10 F1 Scores by Model + Prompt Type')
    plt.xlabel('Model + Prompt Type')
    plt.ylabel('F1 Score')
    plt.xticks(range(len(plot_data)), plot_data['model_prompt'], rotation=45, ha='right')
    
    # Create legend with solid background
    legend = plt.legend(['Top 10', 'Bottom 10'], bbox_to_anchor=(0, 0), loc='lower left', fontsize=7.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    save_plot('top_bottom_f1_scores.png')

def plot_metric_vs_variable(df, x_var, x_label, output_filename):
    """Generic function for plotting F1 score against a variable."""
    create_plot_base()
    
    # Define markers for each prompt type
    markers = {'standard': 'o', 'zero-shot': 's', 'few-shot': '^'}
    
    # Plot each prompt type
    for prompt_type, marker in markers.items():
        mask = df['prompt_type'] == prompt_type
        data = df[mask]
        plt.scatter(data[x_var], data['f1_score'],
                   marker=marker, s=200, label=prompt_type)
    
    # Add labels for each point
    for _, row in df.iterrows():
        plt.annotate(row['model_name_clean'],
                    (row[x_var], row['f1_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title(f'F1 Score vs {x_label} (All Models)', fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    
    # Create legend with solid background
    legend = plt.legend(fontsize=12, bbox_to_anchor=(0, 0), loc='lower left')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    save_plot(output_filename)

def plot_f1_vs_duration(df):
    """Plot F1 scores vs API duration."""
    plot_metric_vs_variable(df, 'api_total_duration_sec', 'API Total Duration (seconds)', 'f1_vs_duration.png')

def plot_f1_vs_tokens(df):
    """Plot F1 scores vs total tokens."""
    plot_metric_vs_variable(df, 'total_tokens', 'Total Tokens', 'f1_vs_tokens.png')

def main():
    """Main function to execute all visualizations."""
    df = load_and_preprocess_data()
    plot_f1_by_model_prompt(df)
    plot_accuracy_by_model_prompt(df)
    plot_top_bottom_f1_scores(df)
    plot_f1_vs_duration(df)
    plot_f1_vs_tokens(df)

if __name__ == "__main__":
    main()