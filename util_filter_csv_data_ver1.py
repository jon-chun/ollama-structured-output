import os
import pandas as pd

def create_output_directory(output_dir):
    """
    Creates the output directory if it doesn't exist.
    
    Args:
        output_dir (str): Path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def load_csv_data(input_path):
    """
    Loads CSV data from the specified path using pandas.
    
    Args:
        input_path (str): Full path to the input CSV file
    
    Returns:
        pandas.DataFrame: Loaded CSV data
    
    Raises:
        FileNotFoundError: If the input file doesn't exist
    """
    try:
        data = pd.read_csv(input_path)
        print(f"Successfully loaded data from {input_path}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")

def filter_and_save_data(data, feature_sets, output_dir):
    """
    Filters the data according to each feature set and saves to separate CSV files.
    
    Args:
        data (pandas.DataFrame): Input DataFrame containing all features
        feature_sets (dict): Dictionary mapping set numbers to lists of feature names
        output_dir (str): Directory where filtered CSV files will be saved
    """
    for set_num, features in feature_sets.items():
        # Verify all requested features exist in the dataset
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            print(f"Warning: Features {missing_features} not found in dataset for set {set_num}")
            continue
            
        # Filter the DataFrame to include only the specified features
        filtered_data = data[features]
        
        # Create output filename based on set number and feature count
        output_filename = f'vignettes_filtered_{set_num}_features-{len(features)}.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the filtered data
        filtered_data.to_csv(output_path, index=False)
        print(f"Created filtered dataset {output_filename} with {len(features)} features")

def main():
    # Define constants
    INPUT_SUBDIR = os.path.join('data')
    INPUT_DATAFILE = 'vignettes_renamed_clean.csv'
    OUTPUT_DIR = os.path.join('data') # ('filtered_data')
    
    # Define feature sets for filtering
    FILTERED_OUTPUT_FILES_DT = {
        0: [
            'sex_recorded_in_1997',
            'marriage_or_cohabitation_status_in_2002',
            'jobs_held_in_2002',
            'household_size_in_1997',
            'parent_or_guardian_relationship_at_age_12',
            'target'
        ],
        1: [
            'sex_recorded_in_1997',
            'marriage_or_cohabitation_status_in_2002',
            'jobs_held_in_2002',
            'household_size_in_1997',
            'parent_or_guardian_relationship_at_age_12',
            'homeless_for_2+_nights_over_last_5_years',
            'used_cocaine_or_other_hard_drug_in_last_4_years',
            'number_of_arrests_in_last_5_years',
            'weight_in_lbs',
            'height_total_inches',
            'target'
        ]
    }
    
    try:
        # Create full input path
        input_path = os.path.join(INPUT_SUBDIR, INPUT_DATAFILE)
        
        # Create output directory
        create_output_directory(OUTPUT_DIR)
        
        # Load the data
        data = load_csv_data(input_path)
        
        # Filter and save data according to feature sets
        filter_and_save_data(data, FILTERED_OUTPUT_FILES_DT, OUTPUT_DIR)
        
        print("Data filtering completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()