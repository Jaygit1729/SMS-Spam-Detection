import nltk
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import joblib
import os
import numpy as np
import sys
from data_preprocessing import transform_text


# Define paths

model_path = "data/model_optimization/best_naive_bayes.pkl"
vectorizer_path = "data/model_optimization/optimized_vectorizer.pkl"
selector_path = "data/model_optimization/feature_selector.pkl"

# Load Model, Vectorizer, and Feature Selector
try:
    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(selector_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        feature_selector = joblib.load(selector_path)
    else:
        st.warning("🔍 Missing files! Ensure the model, vectorizer, and feature selector are in the correct directory.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading files: {str(e)}")
    st.stop()

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")
st.write("Enter a message below, and the model will predict if it's spam or not.")

# User input
input_sms = st.text_area("✍️ Enter the message:", height=100)

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Preprocess text
        processed_text = transform_text(input_sms)

        # Vectorize text
        input_vectorized = vectorizer.transform([processed_text]).toarray()

        # Apply feature selection (ensuring same features as training)
        input_selected = feature_selector.transform(input_vectorized)

        # Make prediction
        prediction = model.predict(input_selected)[0]
        prediction_proba = model.predict_proba(input_selected)[0]

        # Display result

        if prediction == 1:
            st.error(f"🚨 Spam Alert! (Confidence: {prediction_proba[1] * 100:.2f}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {prediction_proba[0] * 100:.2f}%)")
