import pandas as pd
import numpy as np
import os
from logger_utils import setup_logger 

# Initialize Logger for Data Ingestion

logger = setup_logger("logs/data_ingestion.log")
logger.info("Logging setup for data ingestion successfully.")

def load_data(file_path):

    """Loads the dataset from the CSV file."""

    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        logger.info(f'Data Loaded Successfully from {file_path}')
        return df
    
    except Exception as e:
        logger.error(f'Error Loading Data from {file_path}: {e}')
        return None

def save_data(df, output_path):

    """Saves the dataset to the specified output path."""

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f'Data Saved Successfully to {output_path}')

    except Exception as e:
        logger.error(f'Error saving data to {output_path}: {e}')

if __name__ == "__main__":
    
    df = load_data(r"data/UCI_ML/Spam.csv")
    if df is not None:
        save_data(df, r"data/raw_data/spam.csv")
    print(df.head())
