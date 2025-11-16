# logistic_regression.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# -------------------------------
# Step 1: Ensure directories exist
# -------------------------------
os.makedirs("data/training", exist_ok=True)
os.makedirs("data/testing", exist_ok=True)

# -------------------------------
# Step 2: Load train/test splits
# -------------------------------
X_train = pd.read_csv("data/training/X_train.csv")
X_test = pd.read_csv("data/testing/X_test.csv")
y_train = pd.read_csv("data/training/y_train.csv").squeeze()  # convert to Series
y_test = pd.read_csv("data/testing/y_test.csv").squeeze()        # convert to Series

# -------------------------------
# Step 3: Initialize and train Logistic Regression
# -------------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# -------------------------------
# Step 4: Make predictions
# -------------------------------
y_pred = logreg.predict(X_test)

# -------------------------------
# Step 5: Evaluate performance
# -------------------------------
print("=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Step 6: Inspect feature importance
# -------------------------------
coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": logreg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n=== Feature Coefficients ===")
print(coefficients)
