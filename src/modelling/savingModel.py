# src/modelling/xgBoost_Tuned.py
import os
import pickle
import pandas as pd
from xgboost import XGBClassifier

# -------------------------------
# Step 1: Ensure directories exist
# -------------------------------
os.makedirs("models", exist_ok=True)  # create models folder at project root

# -------------------------------
# Step 2: Load train/test splits
# -------------------------------
X_train = pd.read_csv("data/training/X_train.csv")
y_train = pd.read_csv("data/training/y_train.csv").squeeze()

# -------------------------------
# Step 3: Initialize and train XGBoost Tuned
# -------------------------------
xgb = XGBClassifier(
    n_estimators=418,
    max_depth=3,
    learning_rate=0.054811739032681106,
    subsample=0.9272319927266217,
    colsample_bytree=0.7776423796387958,
    min_child_weight=9,
    gamma=1.4254663341802574,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

# -------------------------------
# Step 4: Save the trained model
# -------------------------------
model_path = os.path.join("models", "xgboost_tuned.pkl")
with open(model_path, "wb") as f:
    pickle.dump(xgb, f)

print(f"XGBoost model saved at {model_path}")
