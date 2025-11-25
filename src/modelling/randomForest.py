# random_forest.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Load train/test
X_train = pd.read_csv("data/training/X_train.csv")
X_test = pd.read_csv("data/testing/X_test.csv")
y_train = pd.read_csv("data/training/y_train.csv").squeeze()
y_test = pd.read_csv("data/testing/y_test.csv").squeeze()

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate
print("=== Accuracy ===")
print(accuracy_score(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Feature Importances ===")
print(importances)
