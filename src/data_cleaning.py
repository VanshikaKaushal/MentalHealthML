# src/data_cleaning.py

import pandas as pd

def load_data(path="data/raw/student_depression_dataset.csv"):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


    print("✅ Dataset loaded successfully")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n Shape:", df.shape)
    
    print("\n Info:")
    print(df.info())
    
    print("\n Missing values per column:")
    print(df.isnull().sum())


def clean_columns(df):
    """Cleans and prepares columns for analysis."""
    # Drop unnecessary columns
    drop_cols = ['id', 'Profession', 'Job Satisfaction', 'Work Pressure', 'City']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], axis=1)
    
    # Clean column names (remove spaces, special characters)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(' ', '_')
        .str.replace("'", '', regex=False)
        .str.replace('?', '', regex=False)
        .str.replace('/', '_')
    )
    
    
    return df

def clean_sleep_duration(df):
    if 'Sleep_Duration' in df.columns:
        # Remove quotes, lowercase, strip spaces
        df['Sleep_Duration'] = df['Sleep_Duration'].astype(str).str.replace("'", "").str.lower().str.strip()

        # Remove rows with invalid entries
        df = df[~df['Sleep_Duration'].str.contains('other', na=False)]

        # Standardize the text labels
        df['Sleep_Duration'] = df['Sleep_Duration'].replace({
            'less than 5 hours': '<5',
            '5-6 hours': '5-6',
            '6-7 hours': '6-7',
            '7-8 hours': '7-8',
            'more than 8 hours': '>8'
        })

    return df




def clean_values(df):
    
    df = clean_sleep_duration(df)

    return df

def main():
    """Main cleaning pipeline."""
    path = "data/raw/student_depression_dataset.csv"
    
    print(" Loading dataset...")
    df = load_data(path)
    
    print("\n Preview before cleaning:")
    preview_data(df)
    
    print("\n🧹 Cleaning columns...")
    df_cleaned = clean_columns(df)

    
    print("\n Cleaned Columns:")
    print(df_cleaned.columns.tolist())

    print("\n Cleaning values:")
    df_cleaned = clean_values(df_cleaned)
    
    print("\n Saving cleaned dataset...")
    df_cleaned.to_csv("data/cleaned/student_depression_cleaned.csv", index=False)
    
    print("\n Done! Cleaned dataset saved to 'data/processed/student_depression_cleaned.csv'.")


if __name__ == "__main__":
    main()
