import os
import joblib
import pandas as pd
from data_ingestions import load_data
from logger_utils import setup_logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Initialize Logger
logger = setup_logger("logs/model_building.log")
logger.info("Logging setup for model building successfully.")

def vectorize_text(df):
    """Applies Count vectorization to text data."""
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(df['Processed_Text']).toarray()
    
    logger.info("Applied Count vectorization.")
    return X_vectorized, vectorizer

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Trains a given model and returns its evaluation metrics for both train & test sets."""
    
    model.fit(X_train, y_train)

    # Predictions on Train & Test Set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_pred),
        "Test Precision": precision_score(y_test, y_test_pred),
        "Train Recall": recall_score(y_train, y_train_pred),
        "Test Recall": recall_score(y_test, y_test_pred),
        "Train F1-Score": f1_score(y_train, y_train_pred),
        "Test F1-Score": f1_score(y_test, y_test_pred),
    }

    logger.info(f"{model_name} | Train Acc: {metrics['Train Accuracy']:.4f} | "
                f"Test Acc: {metrics['Test Accuracy']:.4f} | "
                f"Train F1: {metrics['Train F1-Score']:.4f} | "
                f"Test F1: {metrics['Test F1-Score']:.4f}")

    return metrics, model  

if __name__ == "__main__":
    
    df = load_data(r'data/pre-processed/spam.csv')

    if df is not None:
        if 'Processed_Text' not in df.columns or 'Target' not in df.columns:
            raise ValueError("Required columns 'Processed_Text' or 'Target' not found in the dataset.")

        # Apply Count vectorization
        X_vectorized, vectorizer = vectorize_text(df)

        # Prepare target variable
        y = df['Target'].values

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        # ----------------- Train Base Model (Naïve Bayes) -----------------
        naive_bayes_model = MultinomialNB(alpha=3, fit_prior=True)
        base_results, nb_model = train_and_evaluate(naive_bayes_model, X_train, X_test, y_train, y_test, "Naïve Bayes")

        # ----------------- Train Other ML Models -----------------
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        performance_results = [base_results]  
        trained_models = {"Naïve Bayes": nb_model}  

        for name, model in models.items():
            metrics, trained_model = train_and_evaluate(model, X_train, X_test, y_train, y_test, name)
            performance_results.append(metrics)
            trained_models[name] = trained_model  

        # Convert results to DataFrame
        performance_df = pd.DataFrame(performance_results).sort_values(by=["Test F1-Score", "Test Precision", "Test Recall"], ascending=[False, False, False])

        # Print Performance Comparison Table
        print("\n🔹 Model Performance Comparison:\n")
        print(performance_df)

        # ----------------- Select the Best Model -----------------
        best_model_name = performance_df.iloc[0]["Model"]
        best_model = trained_models[best_model_name]  

        # ----------------- Save the Final Model -----------------
        model_dir = "data/model_building"
        os.makedirs(model_dir, exist_ok=True)

        final_model_filename = os.path.join(model_dir, "best_model.pkl")
        vectorizer_filename = os.path.join(model_dir, "vectorizer.pkl")
        performance_results_filename = os.path.join(model_dir, "performance_results.pkl")

        joblib.dump(best_model, final_model_filename)
        joblib.dump(vectorizer, vectorizer_filename)
        joblib.dump(performance_results, performance_results_filename)

        logger.info(f"Final model ({best_model_name}) saved as {final_model_filename}")
        logger.info(f"Vectorizer saved as {vectorizer_filename}")
        logger.info(f"Performance results saved as {performance_results_filename}")

        print(f" Final Model: {best_model_name} selected based on Test F1-Score & Precision!")
