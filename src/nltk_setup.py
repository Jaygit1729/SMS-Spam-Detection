import nltk
import shutil

# Remove any old nltk data (force clean installation)
nltk_data_path = "/home/vscode/nltk_data"

try:
    shutil.rmtree(nltk_data_path)
    print("Deleted old NLTK data folder.")
except FileNotFoundError:
    print("NLTK data folder not found. No need to delete.")

# Download fresh NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
