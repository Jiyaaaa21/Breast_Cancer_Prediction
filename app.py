import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import FunctionTransformer


model = joblib.load("model/rf_xgb_model.pkl")
selected_features = joblib.load("model/selected_features.pkl")
scaler = joblib.load("model/scaler.pkl")
log_transformer = FunctionTransformer(np.log1p)

# Streamlit Page Configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🧠 Breast Cancer Prediction App")
st.markdown("Enter the values for the features below to predict if the tumor is **Benign** or **Malignant**.")

# User input form
with st.form("prediction_form"):
    user_inputs = {}
    for feature in selected_features:
        user_inputs[feature] = st.number_input(f"Enter value for {feature}", min_value=0.0, step=0.01)
    
    submitted = st.form_submit_button("🔍 Predict")

# Process and predict
if submitted:
    # Convert to DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Preprocessing
    scaled = scaler.transform(input_df)
    transformed = log_transformer.transform(scaled)

    # Prediction
    prediction = model.predict(transformed)[0]
    prediction_proba = model.predict_proba(transformed)[0]

    diagnosis = "🟢 Benign" if prediction == 0 else "🔴 Malignant"

    # Display results
    st.subheader("📊 Prediction Results:")
    st.write(f"**Diagnosis:** {diagnosis}")
    st.write(f"**Probability - Benign:** {np.round(prediction_proba[0]*100, 2)}%")
    st.write(f"**Probability - Malignant:** {np.round(prediction_proba[1]*100, 2)}%")
