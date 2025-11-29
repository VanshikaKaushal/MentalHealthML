import pickle
import os
import pandas as pd

model_path = os.path.join("models", "xgboost_tuned.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# XGBoost stores names inside its booster
booster = model.get_booster()

feature_names = booster.feature_names
importances = booster.get_score(importance_type="gain")

# Convert to DataFrame for readability
importance_df = (
    pd.DataFrame(list(importances.items()), columns=["feature", "importance"])
    .sort_values(by="importance", ascending=False)
)

print(importance_df)
