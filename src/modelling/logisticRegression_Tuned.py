# logistic_regression_tuned.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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
y_train = pd.read_csv("data/training/y_train.csv").squeeze()
y_test = pd.read_csv("data/testing/y_test.csv").squeeze()

# -------------------------------
# Step 3: Hyperparameter Tuning
# -------------------------------

param_grid = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "saga"],
    "l1_ratio": [0, 0.5, 1]   # only used when penalty="elasticnet"
}

logreg = LogisticRegression(max_iter=3000, random_state=42)

grid_search = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n=== Best Parameters ===")
print(grid_search.best_params_)

print("\n=== Best CV Accuracy ===")
print(grid_search.best_score_)

# -------------------------------
# Step 4: Train final model with best params
# -------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# -------------------------------
# Step 5: Evaluate performance
# -------------------------------
print("\n=== Test Accuracy ===")
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
    "Coefficient": best_model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n=== Feature Coefficients ===")
print(coefficients)
