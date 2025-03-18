📱 SMS Spam Classifier

🚀 About this Project

This project involves building an SMS Spam Classifier using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to accurately distinguish between spam and non-spam messages, achieving high precision and recall.

📊 About the Dataset

The SMS Spam Collection is a dataset containing 5,574 English messages labeled as either ham (legitimate) or spam. Each line has:

v1: Label (ham or spam)

v2: Raw message text

This dataset was gathered from publicly available sources for research purposes.

🛠️ Methodology

1️⃣ Data Ingestion and Cleaning

Dataset: SMS Spam Collection from Kaggle.

Contains 5,572 messages with 5 columns.

Removed: Unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4).

Duplicates: Dropped 403 duplicates.

Null values: None found, ensuring data integrity.

2️⃣ Feature Engineering and Analysis

New Features Created:

📌 Number of characters per message

📌 Number of words per message

📌 Number of sentences per message

- Statistical analysis provided insights into message length distributions and patterns.

3️⃣ Exploratory Data Analysis (EDA)

📌 Target category counts:

87.37% 'ham' messages

12.63% 'spam' messages

📌 KDE Plot: Revealed skewness in message length and sentence count between categories.

📌 Heatmap: Correlation matrix to uncover feature relationships.

4️⃣ Data Preprocessing

 Applied text preprocessing techniques:

🔹 Lowercasing

🔹 Tokenization

🔹 Special characters removal

🔹 Stopword removal

🔹 Stemming

- Transformed target labels: 'ham' → 0, 'spam' → 1
- Generated word frequency scatter plots for spam vs ham messages.

5️⃣ Text Vectorization (Count Vectorizer)

Count Vectorizer: Converted text into numerical format.

Generated: 6,708 features from the messages.

6️⃣ Model Building and Evaluation

- Split data: 80:20 ratio (training : test)

- Models built:

Naive Bayes (base model)

Two additional models for performance comparison

🏅 Naive Bayes Classifier emerged as the best performer with balanced F1-Score & Precision.

✔️ Saved: Final model and vectorizer for future use.

7️⃣ Model Optimization and Saving

Feature selection: Chi-square test

Hyperparameter tuning: RandomizedSearchCV

Final evaluation: Performance checked on test set

💾 Saved:

Optimized Naive Bayes model (via joblib)

Selected feature set

8️⃣ Streamlit Web App

✨ Built an interactive Streamlit web application for real-time SMS spam classification. Users can input messages and get instant classification results.

🔗 [Live Demo Link] - https://sms-spam-detection-ml.streamlit.app/

🎉 Conclusion

The developed SMS spam classifier:

✅ High F1 Score: Differentiates spam vs ham messages effectively.

✅ Best Model: Naive Bayes proved to be the most efficient performer.

✅ Handles large features: Each word is a feature — thousands of features handled smoothly.

✅ Resilient to irrelevant data: Performance remains strong despite noise.

✅ Fast training & prediction times: Even with large datasets.

⚡ Why Naive Bayes?

Simplicity: Works right out of the box.

Minimal tuning needed: Rarely requires hyperparameter adjustments.

Avoids overfitting: Performs consistently.

Handles high-dimensional data: Perfect for text classification.

🚀 Happy Classifying!

