import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
import numpy as np
import joblib

#LOAD DATA 
dataset = pd.read_csv(r"C:\Users\Jyoti\OneDrive\Desktop\ML PROJECTS\Breast_cancer_prediction\data\data.csv")
# Drop unnecessary columns
dataset.drop(columns=["Unnamed: 32", "id"], inplace=True)


# ENCODING 
from sklearn.preprocessing import LabelEncoder 
encoder = LabelEncoder()
dataset['diagnosis'] = encoder.fit_transform(dataset['diagnosis'])

#  VISUALIZATION 
plt.figure(figsize=(20, 15))
sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

sns.countplot(x="diagnosis", data=dataset)
plt.title("Diagnosis count")
plt.show()

#  SPLIT FEATURES 
X = dataset.drop('diagnosis', axis=1)
y = dataset['diagnosis']

#  HANDLE IMBALANCE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

#FEATURE SELECTION 
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
fs_model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
sfs = SequentialFeatureSelector(fs_model, k_features=7, forward=True, verbose=2)
sfs.fit(X, y)

# Get selected feature names
feature_names = dataset.drop(columns=['diagnosis']).columns
selected_indices = list(sfs.k_feature_idx_)  
selected_features= [feature_names[i] for i in selected_indices]
print("Selected Feature Names:", selected_features)
import joblib
joblib.dump(selected_features, 'model/selected_features.pkl')
print("Selected features saved to model/selected_features.pkl ✅") 

X_selected = pd.DataFrame(X, columns=feature_names)[selected_features] 

#  SCALING + TRANSFORM 
from sklearn.preprocessing import StandardScaler, FunctionTransformer
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

transformer = FunctionTransformer(np.log1p)
X = transformer.fit_transform(X_scaled)
import joblib
joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved to model/scaler.pkl ✅")

# SPLIT DATA 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# HYPERPARAMETER TUNING 
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# --- Random Forest Grid Search ---
rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True],
    'class_weight': ['balanced'],
    'criterion': ['entropy']
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, verbose=2, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("Best Random Forest Parameters:", rf_grid.best_params_)

# --- XGBoost Grid Search ---
xgb_params = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
    'reg_alpha': [0, 0.5],
    'reg_lambda': [1, 1.5]
}
xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                        xgb_params, cv=5, verbose=2, n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print("Best XGBoost Parameters:", xgb_grid.best_params_)

#  ENSEMBLE MODEL 
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb)], voting='soft')
ensemble.fit(X_train, y_train)
print("Ensemble model trained successfully! ✅")

#  EVALUATION
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cv_scores = cross_val_score(ensemble, X_train, y_train, cv=StratifiedKFold(n_splits=10))
print("Cross-Validation Accuracy: {:.2f}%".format(np.mean(cv_scores) * 100))

# Predictions & Evaluation
y_pred = ensemble.predict(X_test)
print("Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Train Accuracy: {:.2f}%".format(ensemble.score(X_train, y_train) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()  

print("Selected Features:", sfs.k_feature_names_)
print("Best Random Forest Parameters:", rf_grid.best_params_)
print("Best XGBoost Parameters:", xgb_grid.best_params_)

# Save model
joblib.dump(ensemble, "model/rf_xgb_model.pkl")
print("Model saved to model/rf_xgb_model.pkl ✅")


