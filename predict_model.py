
import joblib
import numpy as np

# Load model
model = joblib.load("model/rf_xgb_model.pkl")
print("Model loaded ✅")

# Dummy example 
example = np.array([[12.45, 15.7, 82.6, 477.1, 0.127, 0.17, 0.157, 0.08, 0.21, 0.06, 0.42]]) 
prediction = model.predict(example)

print("Prediction:", "Malignant" if prediction[0] == 1 else "Benign")


