import os
import pandas as pd

def remove_columns(input_file="dataset/raw/Reviews.csv", 
                   output_file="dataset/processed/Reviews_removed_columns.csv"):
    """
    Remove specified columns from the dataset and save the processed dataset.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the processed CSV file
    """
    df = pd.read_csv(input_file)
    
    columns_to_remove = ['ProductId', 'UserId', 'ProfileName', 
                         'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time']
    
    df_processed = df.drop(columns=columns_to_remove, errors='ignore')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_processed.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")


if __name__ == "__main__":
    remove_columns()