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
    
    plt.title('F1 Scores by Model and Prompt Type (StandardLLM)')
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
    
    plt.title('Accuracy by Model and Prompt Type (StandardLLM)')
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
    
    plt.title('Top 10 and Bottom 10 F1 Scores (StandardLLM)')
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
    print(f"\n=== Debugging plot_metric_vs_variable for {x_var} ===")
    
    # Data validation and cleaning
    if x_var not in df.columns:
        print(f"ERROR: {x_var} not found in dataframe columns!")
        print("Available columns:", df.columns.tolist())
        return
        
    # Create a copy of the dataframe for processing
    plot_df = df.copy()
    print(f"\nInitial dataframe shape: {plot_df.shape}")
    
    # Convert columns to numeric, replacing any errors with NaN
    print(f"\nConverting {x_var} to numeric...")
    print(f"Original {x_var} dtype:", plot_df[x_var].dtype)
    plot_df[x_var] = pd.to_numeric(plot_df[x_var], errors='coerce')
    print(f"New {x_var} dtype:", plot_df[x_var].dtype)
    print(f"NaN values after conversion: {plot_df[x_var].isna().sum()}")
    
    print("\nConverting f1_score to numeric...")
    print("Original f1_score dtype:", plot_df['f1_score'].dtype)
    plot_df['f1_score'] = pd.to_numeric(plot_df['f1_score'], errors='coerce')
    print("New f1_score dtype:", plot_df['f1_score'].dtype)
    print(f"NaN values after conversion: {plot_df['f1_score'].isna().sum()}")
    
    # Remove any rows with NaN values
    initial_len = len(plot_df)
    plot_df = plot_df.dropna(subset=[x_var, 'f1_score'])
    print(f"\nRows removed due to NaN: {initial_len - len(plot_df)}")
    
    # Print summary statistics
    print(f"\nSummary statistics for {x_var}:")
    print(plot_df[x_var].describe())
    print("\nSummary statistics for f1_score:")
    print(plot_df['f1_score'].describe())
    
    create_plot_base()
    
    # Define markers and colors for each prompt type
    markers = {'cot': 'o', 'cot-nshot': 's', 'system1': '^'}
    colors = {'cot': '#4c72b0', 'cot-nshot': '#dd8452', 'system1': '#55a868'}
    
    print("\nPlotting points for each prompt type:")
    # Plot each prompt type with both markers and colors
    for prompt_type in plot_df['prompt_type'].unique():
        print(f"\nProcessing prompt_type: {prompt_type}")
        if prompt_type in markers:
            mask = plot_df['prompt_type'] == prompt_type
            data = plot_df[mask]
            print(f"Number of points for {prompt_type}: {len(data)}")
            if len(data) > 0:
                print("Sample of points being plotted:")
                print(data[[x_var, 'f1_score']].head())
                
            plt.scatter(data[x_var], data['f1_score'],
                       marker=markers.get(prompt_type, 'o'),
                       c=colors.get(prompt_type, '#4c72b0'),
                       s=200, label=prompt_type,
                       alpha=0.7)
        else:
            print(f"WARNING: Unknown prompt_type '{prompt_type}' - skipping")
    
    print("\nAdding point labels...")
    # Add labels for each point
    for _, row in plot_df.iterrows():
        try:
            plt.annotate(row['model_name_clean'],
                        (row[x_var], row['f1_score']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        except Exception as e:
            print(f"Error adding label for point: {row['model_name_clean']}")
            print(f"Error details: {e}")
    
    plt.title(f'F1 Score vs {x_label} (StandardLLM)', fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create legend with solid background
    legend = plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    print("\nSaving plot...")
    plt.tight_layout()
    save_plot(output_filename)
    print("Plot saved successfully")
    
    # Final summary
    print("\nPlotting complete:")
    print(f"Total points plotted: {len(plot_df)}")
    for prompt_type in plot_df['prompt_type'].unique():
        count = len(plot_df[plot_df['prompt_type'] == prompt_type])
        print(f"Points for {prompt_type}: {count}")

def plot_f1_vs_duration(df):
    """Plot F1 scores vs API duration."""
    print("\n=== Debugging plot_f1_vs_duration ===")
    print(f"Input DataFrame shape: {df.shape}")
    print("\nColumns available:", df.columns.tolist())
    
    # Check api_total_duration_sec column
    if 'api_total_duration_sec' not in df.columns:
        print("ERROR: api_total_duration_sec column not found!")
        return
    
    print("\nUnique prompt types:", df['prompt_type'].unique())
    print("\nSummary of api_total_duration_sec:")
    print(df['api_total_duration_sec'].describe())
    print("\nSummary of f1_score:")
    print(df['f1_score'].describe())
    
    # Check for NaN or infinite values
    print("\nNaN values in relevant columns:")
    print(df[['api_total_duration_sec', 'f1_score', 'prompt_type']].isna().sum())
    
    # Check data types
    print("\nData types of relevant columns:")
    print(df[['api_total_duration_sec', 'f1_score', 'prompt_type']].dtypes)
    
    # Print first few rows for inspection
    print("\nFirst few rows of relevant columns:")
    print(df[['model_name_clean', 'prompt_type', 'api_total_duration_sec', 'f1_score']].head())
    
    # Analyze distribution of points per prompt type
    for ptype in df['prompt_type'].unique():
        subset = df[df['prompt_type'] == ptype]
        print(f"\nPrompt type '{ptype}' has {len(subset)} points")
        print(f"Duration range: {subset['api_total_duration_sec'].min()} to {subset['api_total_duration_sec'].max()}")
        print(f"F1 score range: {subset['f1_score'].min()} to {subset['f1_score'].max()}")
    
    plot_metric_vs_variable(df, 'api_total_duration_sec', 'API Total Duration (seconds)', 'f1_vs_duration.png')

def plot_f1_vs_tokens(df):
    """Plot F1 scores vs total tokens."""
    print("\n=== Debugging plot_f1_vs_tokens ===")
    print(f"Input DataFrame shape: {df.shape}")
    print("\nColumns available:", df.columns.tolist())
    
    # Verify total_tokens calculation
    print("\nChecking token columns:")
    token_cols = [col for col in df.columns if 'token' in col.lower()]
    print("Token-related columns found:", token_cols)
    
    if 'api_prompt_eval_count' in df.columns and 'api_eval_count' in df.columns:
        print("\nToken counts before summation:")
        print("api_prompt_eval_count summary:")
        print(df['api_prompt_eval_count'].describe())
        print("\napi_eval_count summary:")
        print(df['api_eval_count'].describe())
    
    if 'total_tokens' not in df.columns:
        print("WARNING: total_tokens column not found. Attempting to calculate...")
        if 'api_prompt_eval_count' in df.columns and 'api_eval_count' in df.columns:
            df['total_tokens'] = df['api_prompt_eval_count'] + df['api_eval_count']
            print("total_tokens column created")
    
    print("\nSummary of total_tokens:")
    print(df['total_tokens'].describe())
    print("\nSummary of f1_score:")
    print(df['f1_score'].describe())
    
    # Check for NaN or infinite values
    print("\nNaN values in relevant columns:")
    print(df[['total_tokens', 'f1_score', 'prompt_type']].isna().sum())
    
    # Check data types
    print("\nData types of relevant columns:")
    print(df[['total_tokens', 'f1_score', 'prompt_type']].dtypes)
    
    # Print first few rows for inspection
    print("\nFirst few rows of relevant columns:")
    print(df[['model_name_clean', 'prompt_type', 'total_tokens', 'f1_score']].head())
    
    # Analyze distribution of points per prompt type
    for ptype in df['prompt_type'].unique():
        subset = df[df['prompt_type'] == ptype]
        print(f"\nPrompt type '{ptype}' has {len(subset)} points")
        print(f"Token range: {subset['total_tokens'].min()} to {subset['total_tokens'].max()}")
        print(f"F1 score range: {subset['f1_score'].min()} to {subset['f1_score'].max()}")
    
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