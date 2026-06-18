import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Customer_Churn.csv")

# =========================
# 1. Basic Information
# =========================

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nChurn Count:")
print(df['Churn'].value_counts())


# =========================
# 2. How Many Customers Churned?
# =========================

df['Churn'].value_counts().plot(
    kind='bar'
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()


# =========================
# 3. Which Contract Type Churns Most?
# =========================

contract_churn = pd.crosstab(
    df['Contract'],
    df['Churn']
)

print("\nContract vs Churn")
print(contract_churn)

contract_churn.plot(
    kind='bar',
    figsize=(8,5)
)

plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.show()


# =========================
# 4. Does Tenure Affect Churn?
# =========================

plt.figure(figsize=(8,5))

plt.boxplot([
    df[df['Churn']=='No']['tenure'],
    df[df['Churn']=='Yes']['tenure']
])

plt.xticks([1,2], ['No Churn', 'Churn'])
plt.ylabel("Tenure")
plt.title("Tenure vs Churn")

plt.show()


# =========================
# 5. Do High Charges Increase Churn?
# =========================

plt.figure(figsize=(8,5))

plt.boxplot([
    df[df['Churn']=='No']['MonthlyCharges'],
    df[df['Churn']=='Yes']['MonthlyCharges']
])

plt.xticks([1,2], ['No Churn', 'Churn'])
plt.ylabel("Monthly Charges")
plt.title("Monthly Charges vs Churn")

plt.show()
# Check missing values
print(df.isnull().sum())

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Check missing values again
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Remove customerID column
df.drop('customerID', axis=1, inplace=True)

# Convert Churn to 0 and 1
df['Churn'] = df['Churn'].map({
    'No': 0,
    'Yes': 1
})

# Check dataset after preprocessing
print(df.head())

print("New Shape:", df.shape)

print(df.head())
print("New Shape:", df.shape)

# Convert all categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

print(df.head())
print("Encoded Shape:", df.shape)

print(df.dtypes)

# =====================================
# PHASE 4 - MACHINE LEARNING MODEL
# =====================================

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------
# 1. Separate Features and Target
# -------------------------------------

X = df.drop('Churn', axis=1)
y = df['Churn']

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# -------------------------------------
# 2. Train-Test Split
# -------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# -------------------------------------
# 3. Apply SMOTE
# -------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# -------------------------------------
# 4. Train Logistic Regression
# -------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(
    X_train_smote,
    y_train_smote
)

print("\nModel Trained Successfully!")

# -------------------------------------
# 5. Make Predictions
# -------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------
# 6. Evaluate Model
# -------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================")
print("MODEL PERFORMANCE")
print("==========================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

rf_pred = rf_model.predict(X_test)

print("\n===== RANDOM FOREST =====")

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
