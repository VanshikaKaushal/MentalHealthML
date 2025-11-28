import pandas as pd

X_train = pd.read_csv("data/training/X_train.csv")
X_test = pd.read_csv("data/testing/X_test.csv")

print("Any '?' in X_train:", (X_train == '?').any().any())
print("Any '?' in X_test:", (X_test == '?').any().any())
