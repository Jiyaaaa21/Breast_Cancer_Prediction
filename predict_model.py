import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import FunctionTransformer

# Load trained model, selected features, and scaler
model = joblib.load("model/rf_xgb_model.pkl")
selected_features = joblib.load("model/selected_features.pkl") 
scaler = joblib.load("model/scaler.pkl")

# Get user input for each selected feature
print("📥 Please enter the following details:")
input_data = {}
for feature in selected_features:
    while True:
        try:
            value = float(input(f"🔹 Enter value for {feature}: "))
            input_data[feature] = value
            break
        except ValueError:
            print("❌ Please enter a valid number.")

# Create DataFrame from user input
user_df = pd.DataFrame([input_data])

# Scale + Log transform
X_scaled = scaler.transform(user_df)
log_transformer = FunctionTransformer(np.log1p)
X_transformed = log_transformer.transform(X_scaled)

# Make prediction
prediction = model.predict(X_transformed)[0]
prediction_proba = model.predict_proba(X_transformed)

# Output result
diagnosis = "🔴 Malignant" if prediction == 1 else "🟢 Benign"
print(f"\n✅ Predicted Diagnosis: {diagnosis}")
print(f"📊 Prediction Probabilities (Benign, Malignant): {np.round(prediction_proba[0], 3)}")
