# src/data_cleaning.py
import math
import pandas as pd

def load_data(path="data/raw/student_depression_dataset.csv"):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(path)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n Shape:", df.shape)
    
    print("\n Info:")
    print(df.info())
    
    print("\n Missing values per column:")
    print(df.isnull().sum())
    return df

def create_DegreeCategory(df):
    """
    Creates a new column 'Degree_Category' based on the 'Degree' column.
    Groups degrees into broader categories like STEM, Arts, Business, Graduate, High School, PhD.
    """
    if 'Degree' in df.columns:
        # Standardize degree text
        df['Degree'] = df['Degree'].astype(str).str.lower().str.strip().str.replace("'", "")

        # Mapping from specific degrees to broader categories
        degree_mapping = {
            "b.arch": "STEM_Bach",
            "b.com": "Business",
            "b.ed": "Arts",
            "class 12": "HighSchool",
            "b.pharm": "STEM_Bach",
            "b.tech": "STEM_Bach",
            "b.a": "Arts",
            "bba": "Business",
            "bca": "STEM_Bach",
            "be": "STEM_Bach",
            "bhm": "Arts",
            "bsc": "STEM_Bach",
            "llb": "Arts",
            "llm": "Graduate",
            "m.com": "Graduate",
            "m.ed": "Graduate",
            "m.pharm": "Graduate",
            "m.tech": "Graduate",
            "ma": "Graduate",
            "mba": "Graduate",
            "mbbs": "STEM",
            "mca": "Graduate",
            "md": "Graduate",
            "me": "Graduate",
            "mhm": "Graduate",
            "mac": "Graduate",
            "phd": "PhD"
        }

        # Create the new column and fill unmapped values with 'Other'
        df['Degree_Category'] = df['Degree'].map(degree_mapping).fillna("Other")

    return df


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

def clean_degree(df):
    if 'Degree' in df.columns:
        # Standardize text
        df['Degree'] = df['Degree'].astype(str).str.lower().str.strip()

        # Remove rows where Degree is "other" or "others"
        df = df[~df['Degree'].str.contains('other', na=False)]

        
    return df


def clean_dietary_habits(df):
    if 'Dietary_Habits' in df.columns:
        # Standardize text: lowercase and strip spaces
        df['Dietary_Habits'] = df['Dietary_Habits'].astype(str).str.lower().str.strip()

        # Remove rows where Dietary_Habits is "other" or "others"
        df = df[~df['Dietary_Habits'].str.contains('other', na=False)]

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

def clean_CGPA(df):
    if 'CGPA' in df.columns:
        # Remove quotes, strip spaces
        df['CGPA'] = df['CGPA'].astype(str).str.replace("'", "").str.strip()

        # Remove rows with invalid entries (non-numeric)
        df = df[df['CGPA'].str.replace('.', '', 1).str.isnumeric()]

        # Convert to float
        df['CGPA'] = df['CGPA'].astype(float)

        # Round CGPA to nearest integer
        df['CGPA'] = df['CGPA'].apply(lambda x: math.floor(x) + 1 if (x - math.floor(x)) >= 0.5 else math.floor(x))

    return df


def clean_values(df):
    
    df = clean_sleep_duration(df)

    df = clean_CGPA(df)

    df = clean_dietary_habits(df)
    df = clean_degree(df)
    df = create_DegreeCategory(df)


    return df

def main():
    """Main cleaning pipeline."""
    path = "data/raw/student_depression_dataset.csv"
    
    print(" Loading dataset...")
    df = load_data(path)
    
    
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
