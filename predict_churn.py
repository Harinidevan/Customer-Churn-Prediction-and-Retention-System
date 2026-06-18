import joblib
import pandas as pd
from churn_explanation import explain_churn
from retention_engine import get_recommendations

# Load model
model = joblib.load("customer_churn_model.pkl")

print("Model loaded successfully!")

# Sample customer data
customer = pd.DataFrame([{
    'SeniorCitizen': 0,
    'tenure': 5,
    'MonthlyCharges': 95,
    'TotalCharges': 500,
    'gender_Male': 1,
    'Partner_Yes': 0,
    'Dependents_Yes': 0,
    'PhoneService_Yes': 1,
    'MultipleLines_No phone service': 0,
    'MultipleLines_Yes': 1,
    'InternetService_Fiber optic': 1,
    'InternetService_No': 0,
    'OnlineSecurity_No internet service': 0,
    'OnlineSecurity_Yes': 0,
    'OnlineBackup_No internet service': 0,
    'OnlineBackup_Yes': 0,
    'DeviceProtection_No internet service': 0,
    'DeviceProtection_Yes': 0,
    'TechSupport_No internet service': 0,
    'TechSupport_Yes': 0,
    'StreamingTV_No internet service': 0,
    'StreamingTV_Yes': 1,
    'StreamingMovies_No internet service': 0,
    'StreamingMovies_Yes': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'PaperlessBilling_Yes': 1,
    'PaymentMethod_Credit card (automatic)': 0,
    'PaymentMethod_Electronic check': 1,
    'PaymentMethod_Mailed check': 0
}])

# Predict
prediction = model.predict(customer)

# Probability
probability = model.predict_proba(customer)

print("\nPrediction Result")

if prediction[0] == 1:

    print("Customer Will Churn")

    reasons = explain_churn(customer.iloc[0])

    print("\nReasons for Churn:")

    for reason in reasons:
        print("-", reason)

    recommendations = get_recommendations(reasons)

    print("\nRetention recommendations:")

    for rec in recommendations:
        print("-", rec)

else:
    print("Customer Will Stay")

print(f"\nChurn Probability: {probability[0][1]*100:.2f}%")