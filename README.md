# 📊 Customer Churn Prediction & Retention System

## 📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. This project predicts whether a customer is likely to leave a service and provides explanations and retention recommendations.

The system uses Machine Learning to identify high-risk customers and helps businesses take proactive actions to improve customer retention.

---

## 🎯 Project Objectives

* Predict whether a customer will churn.
* Explain the reasons behind churn.
* Provide retention recommendations.
* Visualize churn-related insights.
* Offer an interactive Streamlit web application.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* Joblib
* Streamlit

---

## 📂 Dataset

Dataset: Customer Churn Dataset

The dataset contains customer information such as:

* Customer demographics
* Contract type
* Internet service details
* Payment methods
* Monthly charges
* Total charges
* Customer tenure
* Churn status

---

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Churn Distribution
* Contract Type vs Churn
* Tenure vs Churn
* Monthly Charges vs Churn

Visualizations were created using Matplotlib.

---

## ⚙️ Data Preprocessing

Steps performed:

1. Handled missing values.
2. Converted TotalCharges to numeric.
3. Removed unnecessary columns.
4. Encoded categorical variables using One-Hot Encoding.
5. Balanced the dataset using SMOTE.
6. Split data into training and testing sets.

---

## 🤖 Machine Learning Model

Model Used:

* Random Forest Classifier

Evaluation:

* Trained on balanced data using SMOTE.
* Feature Importance Analysis performed.
* Model saved using Joblib.

Saved Model:

customer_churn_model.pkl

---

## 🔍 Feature Importance

Top factors affecting customer churn:

* Total Charges
* Tenure
* Monthly Charges
* Electronic Check Payment Method
* Fiber Optic Internet Service
* Paperless Billing
* Contract Type

---

## 🚀 Key Features

### 1. Churn Prediction

Predicts whether a customer will churn.

### 2. Churn Explanation Engine

Explains the reasons behind churn such as:

* Low tenure
* High monthly charges
* Month-to-month contract
* Electronic check payment method

### 3. Retention Recommendation Engine

Provides suggestions such as:

* Loyalty discounts
* Long-term contract incentives
* Customer onboarding support
* Automatic payment options

### 4. Streamlit Dashboard

Interactive web application for:

* Customer input
* Churn prediction
* Risk analysis
* Recommendations

---

## 📈 Risk Classification

The application classifies customers as:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

based on churn probability.

---

## 📁 Project Structure

Customer_Churn_Prediction/

├── app.py

├── churn_analysis.py

├── feature_importance.py

├── churn_explanation.py

├── retention_engine.py

├── predict_churn.py

├── customer_churn_model.pkl

├── Customer_Churn.csv

├── requirements.txt

└── README.md

---

## ▶️ How to Run

### Install Dependencies

pip install -r requirements.txt

### Run Streamlit Application

streamlit run app.py

---

## 📌 Future Enhancements

* AI Chatbot Integration
* Advanced Explainable AI (XAI)
* Real-Time Customer Monitoring
* Cloud Deployment
* Interactive Business Dashboard

---

## 👩‍💻 Author

Harini D

AI & Data Science Student

St. Joseph's College of Engineering

---

## ⭐ Project Outcome

This project not only predicts customer churn but also explains the reasons behind churn and provides actionable retention strategies, helping businesses improve customer satisfaction and reduce customer loss.
