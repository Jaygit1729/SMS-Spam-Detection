import joblib
import os
import json
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logger_utils import setup_logger
from data_ingestions import load_data
from sklearn.feature_selection import SelectKBest, chi2

# Setup logging
logger = setup_logger("logs/model_optimization.log")
logger.info("Logging setup for model optimization successfully.")

def load_vectorizer():
    """Loads the saved Count Vectorizer to ensure consistency."""
    
    vectorizer_path = "data/model_building/vectorizer.pkl"
    if os.path.exists(vectorizer_path):
        return joblib.load(vectorizer_path)
    logger.warning("Vectorizer not found! Ensure you have run the model building pipeline.")
    return None

def feature_selection(X_train, y_train, X_test, y_test, k_values=[3000, 4000, 5000]):
    """Selects the best k-value based on F1-score."""
    
    best_k, best_f1, best_X_train, best_X_test, best_selector = None, 0, None, None, None

    for k in k_values:
        selector = SelectKBest(chi2, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Quick evaluation with baseline Naïve Bayes model
        model = MultinomialNB(alpha=1)
        model.fit(X_train_selected, y_train)
        f1 = f1_score(y_test, model.predict(X_test_selected))

        logger.info(f"Feature selection with k={k} | F1-score: {f1:.4f}")

        if f1 > best_f1:
            best_k, best_f1 = k, f1
            best_X_train, best_X_test = X_train_selected, X_test_selected
            best_selector = selector

    logger.info(f"Selected best k={best_k} with F1-score: {best_f1:.4f}")
    return best_X_train, best_X_test, best_selector

def optimize_naive_bayes(X_train, y_train):
    """Performs hyperparameter tuning for Naïve Bayes using RandomizedSearchCV."""
    
    param_grid = {
        'alpha': [0.1, 0.5, 1, 2, 3, 5, 10],  
        'fit_prior': [True, False],
    }
    
    model = MultinomialNB()
    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='f1', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)

    logger.info(f"Best parameters for Naïve Bayes: {search.best_params_}")
    return search.best_estimator_, search.best_params_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates a model and returns performance metrics."""
    
    y_pred = model.predict(X_test)
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    
    logger.info(f"{model_name} | Accuracy: {metrics['Accuracy']:.4f} | "
                f"Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | "
                f"F1-score: {metrics['F1-Score']:.4f}")
    
    return metrics

if __name__ == "__main__":
    logger.info("Starting Naïve Bayes model optimization...")

    df = load_data(r'data/pre-processed/spam.csv')
    if df is None:
        raise ValueError("Data loading failed!")

    vectorizer = load_vectorizer()
    if vectorizer is None:
        raise ValueError("Vectorizer not found. Re-run model_building.py to generate it.")

    X_vectorized = vectorizer.transform(df['Processed_Text']).toarray()
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Feature selection (automatically finds the best k)
    X_train_selected, X_test_selected, selector = feature_selection(X_train, y_train, X_test, y_test)

    # Save the best feature selector
    joblib.dump(selector, "data/model_optimization/feature_selector.pkl")

    # Hyperparameter tuning
    best_model, best_params = optimize_naive_bayes(X_train_selected, y_train)

    # Evaluate and save model
    optimized_results = evaluate_model(best_model, X_test_selected, y_test, "Optimized Naïve Bayes")

    model_dir = "data/model_optimization"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(model_dir, "best_naive_bayes.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "optimized_vectorizer.pkl"))

    with open(os.path.join(model_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f)

    logger.info("Optimized Naïve Bayes model, vectorizer, and parameters saved successfully.")

    print("\n Best Hyperparameters:", best_params)
    print("\n Optimized Model Performance:")
    print(pd.DataFrame([optimized_results]))
    print("\n Model optimization completed successfully!")
