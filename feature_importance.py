import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("Customer_Churn.csv")

# ==========================
# PREPROCESSING
# ==========================

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Remove missing values
df.dropna(inplace=True)

# Remove customerID
df.drop('customerID', axis=1, inplace=True)

# Convert Churn column to 0 and 1
df['Churn'] = df['Churn'].map({
    'No': 0,
    'Yes': 1
})

# Convert categorical columns to numeric
df = pd.get_dummies(
    df,
    drop_first=True
)

# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop('Churn', axis=1)
y = df['Churn']

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# SMOTE
# ==========================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# ==========================
# RANDOM FOREST MODEL
# ==========================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

# ==========================
# TOP 10 FEATURES
# ==========================

print("\nTop 10 Features Affecting Churn:\n")
print(feature_importance.head(10))

# ==========================
# VISUALIZATION
# ==========================

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.title("Top Factors Affecting Customer Churn")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.show()

import joblib
joblib.dump(rf_model,"customer_churn_model.pkl")
print("model saved successfully!")
print(X.columns.tolist())