# churn_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

data = pd.read_csv("data/customer_churn.csv")

if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric (sometimes it has spaces)
if 'TotalCharges' in data.columns:
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median()
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

X = data.drop("Churn", axis=1)   # Features
y = data["Churn"]                # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved as models/churn_model.pkl")
print("Scaler saved as models/scaler.pkl")

def predict_churn(new_data):
    """
    new_data: dict of customer features
    Example:
    predict_churn({
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        ...
    })
    """
    df = pd.DataFrame([new_data])

# Example Test
# new_customer = {
#     "gender": "Female",
#     "SeniorCitizen": 0,
#     "Partner": "Yes",
#     "Dependents": "No",
#     "tenure": 12,
#     "PhoneService": "Yes",
#     "MultipleLines": "No",
#     "InternetService": "Fiber optic",
#     "OnlineSecurity": "No",
#     "OnlineBackup": "Yes",
#     "DeviceProtection": "No",
#     "TechSupport": "No",
#     "StreamingTV": "Yes",
#     "StreamingMovies": "No",
#     "Contract": "Month-to-month",
#     "PaperlessBilling": "Yes",
#     "PaymentMethod": "Electronic check",
#     "MonthlyCharges": 70.35,
#     "TotalCharges": 1397.5
# }
# print(predict_churn(new_customer))
