About this project

SMS-Spam-Classifier:
 

This project involves building an SMS spam classifier using NLP and Machine Learning techniques. The classifier accurately distinguishes between spam and non-spam messages, achieving high precision and recall.
 
About Dataset:
 

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains SMS messages in English of 5,574 messages, tagged as ham (legitimate) or spam. 
 

The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
 

This corpus has been collected from free or free for research sources at the Internet.
 

Methodology:
 

1. Data Ingestions and Cleaning
 

The SMS Spam Collection Dataset from Kaggle was used for this project.
 

The dataset contains 5,572 messages with 5 columns.
 

Irrelevant columns (Unnamed: 2, Unnamed: 3, Unnamed: 4) were removed.
 

Duplicates (403) were dropped from the dataset.
 

The dataset contained no null values, ensuring data integrity.
 

2. Feature Engineering and Analysis:
 

Three New Features has been created to find out : A)  Number of characters in each SMS B) Number of words in each SMS and C) Number of sentences in each SMS
 

Statistical summaries revealed insights on message length distribution and other aspects of the data.
 

3. EDA and Data Visualization
 

The value counts of target categories ('ham' and 'spam') were calculated.
 

The distribution of target categories showed that 87.37% were 'ham' messages and 12.63% were 'spam' messages.
 

KDE Plot revealed the presence of skewness for number of characters and number of sentences in both categories.
 

Generates a heatmap for the correlation matrix to see the relationship of features with each other.
 

4. Data Preprocessing
 

Text preprocessing techniques such as lower casing, tokenization, removing special characters, stopword removal, and stemming were applied to the messages.
 

Transforms the target column: 'ham' → 0, 'spam' → 1.
 

Generates a word frequency scatter plot for spam or ham messages to see the frequently occuring word in a corpus.
 

5. Text Vectorization Using Count Vectorization
 

Count vectorization was performed on both the training and test sets to convert text data into numerical form.
 

A total of 6708 features were generated using Count Vectorization. 
 
6. Model Building and Evaluation
 

The data was split into training and test sets in a 80:20 ratio.
 

Created Naive Bayes as a base model and two other model were also implemented to compare the performance of the model. Naive Bayes Classifier demonstrated excellent performance with balanced F1-Score & Precision.
 

Saved the Final model and Vectorizer for future use.
 

7. Model Optimization and Saving
 

Performs feature selection using the chi-square test.
 

Optimizes Naïve Bayes using RandomizedSearchCV.
 

Evaluates model performance on the test set.
 

Saves:
Optimized Naive Bayes model was saved using the joblib library.
Selected feature set.
 
 

9. Streamlit App 
 

Developed a Streamlit web application to enable real-time SMS spam classification for new messages.

 

Conclusion
 

In conclusion, the developed SMS spam classifier successfully differentiates between spam and ham messages with high accuracy and precision. The Naive Bayes Classifier emerged as the best-performing model, providing consistent results on both training and test data. This project showcases the effectiveness of NLP techniques and machine learning algorithms in tackling text classification tasks.
 

One of the major advantages that Naive Bayes has over other classification algorithms is its ability to handle an extremely large number of features. In our case, each word is treated as a feature and there are thousands of different words. Also, it performs well even with the presence of irrelevant features and is relatively unaffected by them.
 

The other major advantage it has is its relative simplicity. Naive Bayes' works well right out of the box and tuning it's parameters is rarely ever necessary, except usually in cases where the distribution of the data is known.
 

It rarely ever overfits the data.
 

Another important advantage is that its model training and prediction times are very fast for the amount of data it can handle.
 

Demo Link : 