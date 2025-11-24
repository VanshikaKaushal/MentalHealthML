import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("data/encoded/student_depression_encoded.csv")

# Features (X) and target (y)
X = df.drop("Depression", axis=1)
y = df["Depression"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # ensures reproducible split
    stratify=y            # preserves class balance
)

# Features
X_train.to_csv("data/training/X_train.csv", index=False)
X_test.to_csv("data/testing/X_test.csv", index=False)

# Target
y_train.to_csv("data/training/y_train.csv", index=False)
y_test.to_csv("data/testing/y_test.csv", index=False)

##rename this script to split_data.py