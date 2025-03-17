import numpy as np
import pandas as pd
from logger_utils import setup_logger
from data_ingestions import load_data
from data_ingestions import save_data
import nltk
nltk.download('punkt', quiet=True)
import os


# Initialize Logger for Data Cleaning
logger = setup_logger("logs/feature_engineering.log")
logger.info("Logging setup for feature_engineering successfully.")

def num_chara(df):

    """Returns a DataFrame with the number of characters in each SMS."""
    
    df['num_character'] = df['Text'].apply(len)
    return df  

def num_word(df):
    
    """Returns a DataFrame with the number of words in each SMS."""
    
    df['num_words'] = df['Text'].apply(lambda x: len(nltk.word_tokenize(x)))
    return df 

def num_sentences(df):
    
    """Returns a DataFrame with the number of sentences in each SMS."""
    
    df['num_sentences'] = df['Text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    return df

if __name__ == '__main__':
    
    df = load_data(r"data/data_cleaning/cleaned_spam.csv")
    
    if df is not None:
        
        logger.info("Loaded cleaned data for EDA.")
        df = num_chara(df)
        df = num_word(df)
        df = num_sentences(df)

        print(df.sample(5))  

        # Save the updated DataFrame after transformations
        save_data(df, r"data/feature_engineering/spam.csv")

