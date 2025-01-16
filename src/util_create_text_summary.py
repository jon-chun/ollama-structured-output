import os
import pandas as pd

def create_demographic_summary(row):
    """
    Creates a natural language summary of a person's demographic information.
    
    Args:
        row (pandas.Series): A row from the dataset containing demographic information
        
    Returns:
        str: A human-readable summary of the demographic information
    """
    # Initialize empty list to store sentence parts
    summary_parts = []
    
    # Add sex and age information (from 1997)
    sex = row['sex_recorded_in_1997']
    summary_parts.append(f"The person is a {sex}")
    
    # Add marriage/cohabitation status (from 2002)
    relationship_status = row['marriage_or_cohabitation_status_in_2002']
    summary_parts.append(f"who is {relationship_status}")
    
    # Add employment information (from 2002)
    jobs = row['jobs_held_in_2002']
    if jobs == 0:
        summary_parts.append("and is currently unemployed")
    else:
        summary_parts.append(f"and has held {jobs} jobs")
    
    # Add household information (from 1997)
    household_size = row['household_size_in_1997']
    summary_parts.append(f"with a household size of {household_size}")
    
    # Add guardian information
    guardian = row['parent_or_guardian_relationship_at_age_12']
    if guardian != "Unknown":
        summary_parts.append(f"and lived with their {guardian} at age 12")
    
    # Combine all parts into a single sentence
    text_summary = " ".join(summary_parts) + "."
    
    return text_summary

def util_create_text_summary(input_path, output_path):
    """
    Reads a CSV file, creates text summaries for each row, and saves the result
    to a new CSV file with an additional 'text_summary' column.
    
    Args:
        input_path (str): Path to the input CSV file
        output_path (str): Path to save the output CSV file
        
    Returns:
        None
    """
    try:
        # Read the input CSV file
        print(f"Reading input file: {input_path}")
        df = pd.read_csv(input_path)
        
        # Create text summaries for each row
        print("Generating text summaries...")
        df['text_summary'] = df.apply(create_demographic_summary, axis=1)
        
        # Save the updated dataframe to the output file
        print(f"Saving output file: {output_path}")
        df.to_csv(output_path, index=False)
        
        print("Text summary generation completed successfully!")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    # Define input and output file paths
    INPUT_DATAFILE = os.path.join('data', 'vignettes_filtered_0_features-6.csv')
    OUTPUT_DATAFILE = os.path.join('data', 'vignettes_filtered_0_features-6_text_summary.csv')
    
    # Create text summaries
    util_create_text_summary(INPUT_DATAFILE, OUTPUT_DATAFILE)