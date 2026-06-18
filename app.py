import streamlit as st
import pandas as pd
import joblib

from churn_explanation import explain_churn
from retention_engine import get_recommendations

# ==========================
# PAGE SETTINGS
# ==========================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("customer_churn_model.pkl")

# ==========================
# TITLE
# ==========================

st.title("📊 Customer Churn Prediction & Retention System")

st.write(
    "Predict customer churn and get retention recommendations."
)

# ==========================
# USER INPUTS
# ==========================

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=5
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=95.0
)

contract = st.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Credit card",
        "Mailed check"
    ]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    [
        "No",
        "Yes"
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "Fiber optic",
        "No"
    ]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    [
        "No",
        "Yes"
    ]
)

# ==========================
# PREDICT BUTTON
# ==========================

if st.button("Predict"):

    customer = {

        'SeniorCitizen': 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': tenure * monthly_charges,

        'gender_Male': 1,
        'Partner_Yes': 0,
        'Dependents_Yes': 0,
        'PhoneService_Yes': 1,

        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,

        'InternetService_Fiber optic': 0,
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

        'PaperlessBilling_Yes': 0,

        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0
    }

    # ==========================
    # CONTRACT
    # ==========================

    if contract == "One year":
        customer['Contract_One year'] = 1

    elif contract == "Two year":
        customer['Contract_Two year'] = 1

    # ==========================
    # PAYMENT METHOD
    # ==========================

    if payment_method == "Electronic check":
        customer['PaymentMethod_Electronic check'] = 1

    elif payment_method == "Credit card":
        customer['PaymentMethod_Credit card (automatic)'] = 1

    elif payment_method == "Mailed check":
        customer['PaymentMethod_Mailed check'] = 1

    # ==========================
    # MULTIPLE LINES
    # ==========================

    if multiple_lines == "Yes":
        customer['MultipleLines_Yes'] = 1

    # ==========================
    # INTERNET SERVICE
    # ==========================

    if internet_service == "Fiber optic":
        customer['InternetService_Fiber optic'] = 1
    else:
        customer['InternetService_No'] = 1

    # ==========================
    # PAPERLESS BILLING
    # ==========================

    if paperless_billing == "Yes":
        customer['PaperlessBilling_Yes'] = 1

    # ==========================
    # DATAFRAME
    # ==========================

    customer_df = pd.DataFrame([customer])

    # ==========================
    # PREDICTION
    # ==========================

    prediction = model.predict(customer_df)
    probability = model.predict_proba(customer_df)

    risk = probability[0][1] * 100

    # ==========================
    # CHURN PROBABILITY
    # ==========================

    st.metric(
        "Churn Probability",
        f"{risk:.2f}%"
    )

    # ==========================
    # RISK LEVEL
    # ==========================

    st.subheader("Risk Level")

    if risk < 30:
        st.success("🟢 Low Risk")

    elif risk < 70:
        st.warning("🟡 Medium Risk")

    else:
        st.error("🔴 High Risk")

    # ==========================
    # PREDICTION RESULT
    # ==========================

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("Customer Will Churn")

        reasons = explain_churn(customer)

        st.subheader("Reasons for Churn")

        for reason in reasons:
            st.write("•", reason)

        recommendations = get_recommendations(reasons)

        st.subheader("Retention Recommendations")

        for rec in recommendations:
            st.write("•", rec)

    else:

        st.success("Customer Will Stay")

        st.subheader("Positive Factors")

        st.write("• Customer has lower churn risk")
        st.write("• Current service plan appears stable")
        st.write("• Customer is likely to remain subscribed")