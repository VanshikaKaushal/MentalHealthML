# xgboost_model.py

import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# -----------------------
# Load data
# -----------------------
X_train = pd.read_csv("data/training/X_train.csv")
X_test = pd.read_csv("data/testing/X_test.csv")
y_train = pd.read_csv("data/training/y_train.csv").squeeze()
y_test = pd.read_csv("data/testing/y_test.csv").squeeze()

# -----------------------
# Hyperparameter Tuning with Optuna
# -----------------------

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return score.mean()

print("Running Optuna tuning...")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best Parameters:", study.best_trial.params)

# -----------------------
# Train final model with best parameters
# -----------------------
best_params = study.best_trial.params
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
y_pred = best_model.predict(X_test)

print("\n=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# -----------------------
# Feature Importance
# -----------------------
importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Feature Importances ===")
print(importances)
