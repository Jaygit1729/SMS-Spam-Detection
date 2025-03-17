from data_ingestions import load_data, save_data
from logger_utils import setup_logger
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import plotly.express as px
from collections import Counter
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Logger for Data Cleaning
logger = setup_logger("logs/feature_engineering.log")
logger.info("Logging setup for feature_engineering successfully.")

def transform_text(text):

    """ Preprocesses a given text by applying NLP techniques: lowercasing, tokenization, 
    removing special characters, stopword removal, and stemming."""
    
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Tokenization
    tokens = word_tokenize(text)

    # Step 3: Remove special characters and numbers
    tokens = [word for word in tokens if word.isalnum()]

    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Step 5: Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Return processed text as a single string
    processed_text = " ".join(tokens)

    return processed_text if processed_text.strip() else None  # Return None if text is empty

def transform_target(df, target_column):
    
    """ Transforms the target column: 'ham' → 0, 'spam' → 1."""
    
    if target_column in df.columns:
        df[target_column] = df[target_column].map({'ham': 0, 'spam': 1})
        return df
    else:
        raise ValueError(f"Column '{target_column}' not found in the DataFrame.")

def generate_spam_ham_wordcloud(df, text_column, target_column, target_value):
    
    """ Generates a word frequency scatter plot for spam or ham messages."""
    
    # Determine label based on target value
    label = "Spam" if target_value == 1 else "Ham"

    # Filter messages based on target value
    filtered_text = " ".join(df[df[target_column] == target_value][text_column])

    # Compute word frequencies
    word_freq = Counter(filtered_text.split())

    # Convert to DataFrame and sort
    word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
    word_freq_df = word_freq_df.sort_values(by="Frequency", ascending=False).head(30)

    # Create the scatter plot (word cloud-like)
    fig = px.scatter(
        word_freq_df, x=range(len(word_freq_df)), y="Frequency",
        size="Frequency", text="Word",
        title=f"{label} Messages Word Cloud",
        labels={'x': "Words (index)", 'y': "Frequency"}
    )

    # Adjust text position and layout
    fig.update_traces(textposition='top center', textfont_size=12)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        xaxis=dict(tickangle=-45),  # Rotate labels for readability
        height=700, width=1200,  # Set figure size
        margin=dict(l=50, r=50, t=50, b=100)  # Adjust margins
    )

    return fig



if __name__ == "__main__":

    df = load_data(r'data/feature_engineering/spam.csv')
    logger.info(f"Data Loaded Successfully for Data Pre-Processing")
    
    if df is not None:
        # Ensure 'Text' and 'Target' columns exist
        if 'Text' not in df.columns or 'Target' not in df.columns:
            raise ValueError("Required columns 'Text' or 'Target' not found in the dataset.")

        # Apply text preprocessing
        df['Processed_Text'] = df['Text'].apply(transform_text)

        # Drop rows where 'Processed_Text' became empty or null after transformation
        initial_rows = len(df)
        df.dropna(subset=['Processed_Text'], inplace=True)
        removed_rows = initial_rows - len(df)
        logger.info(f"Dropped {removed_rows} rows where 'Processed_Text' became empty/null after processing.")

        # Apply target transformation
        df = transform_target(df, 'Target')

        # Generate and display word cloud for spam
        fig1 = generate_spam_ham_wordcloud(df, text_column='Processed_Text', target_column='Target', target_value=1)
        fig1.show()

        # Generate and display word cloud for ham
        fig2 = generate_spam_ham_wordcloud(df, text_column='Processed_Text', target_column='Target', target_value=0)
        fig2.show()

        # Save preprocessed dataset
        save_data(df, r"data/pre-processed/spam.csv")

    print("Text preprocessing and target transformation completed successfully!")
