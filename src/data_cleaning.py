import pandas as pd
import numpy as np
from data_ingestions import load_data, save_data
from logger_utils import setup_logger  

# Initialize Logger for Data Cleaning

logger = setup_logger("logs/data_cleaning.log")
logger.info("Logging setup for data cleaning successfully.")

def drop_unnecessary_columns(df, columns_to_drop):

    """Drops specified columns from the DataFrame."""
    
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    logger.info(f"Dropped Columns: {columns_to_drop}")
    return df

def drop_duplicate_rows(df):
    
    """Drops duplicate rows from the DataFrame."""
    
    initial_shape = df.shape
    df = df.drop_duplicates(keep='first')
    final_shape = df.shape
    logger.info(f"Dropped {initial_shape[0] - final_shape[0]} duplicate rows")
    return df

def rename_column(df, column_mapping):
    
    """Renames the existing column names for better readability."""
    
    df.rename(columns=column_mapping, inplace=True)
    logger.info(f"Renamed column: {column_mapping}")
    return df

def missing_prct(df):
    
    """Calculates the percentage of missing values for each column in the DataFrame."""
    
    df_missing_prct = (df.isnull().sum() / df.shape[0]) * 100
    logger.info(f"Percentage of Missing values for each feature/target:\n{df_missing_prct.to_string(index=False)}")
    return df

def clean_data(file_path):
    
    df = load_data(file_path)
    if df is None:
        return None
    
    df = drop_unnecessary_columns(df, ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    df = drop_duplicate_rows(df)
    df = rename_column(df, {'v1': 'Target', 'v2': 'Text'})
    df = missing_prct(df)
    logger.info("Data Cleaning Completed Successfully.")
    return df 

if __name__ == "__main__":
    
    cleaned_df = clean_data(r'data/raw_data/spam.csv')
    if cleaned_df is not None:
        save_data(cleaned_df,r'data/data_cleaning/cleaned_spam.csv')
        
