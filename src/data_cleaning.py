# src/exploratory_analysis.py

import pandas as pd

def load_and_preview_data():
    path = "data/raw/student_depression_dataset.csv"
    df = pd.read_csv(path)

    print("✅ Dataset loaded successfully")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:", df.shape)
    print("\nInfo:")
    print(df.info())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    load_and_preview_data()