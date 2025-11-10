# This file encodes our cleaned data so that dataset is all fully numeric and ready for Machine Learning. 

import pandas as pd

def binaryConvert(df, columnName):
    df[columnName] = df[columnName].astype(str).replace({
        'Yes' : 1,
        'No' : 0
    })
    return df

def oneHotEncoding(df, categories, columnName):
    """
    One-hot encode a column based on the list of possible categories.
    """
    for category in categories:
        df[columnName + '_' + category] = (df[columnName] == category).astype(int)
    
    # Drop the original column
    df = df.drop(columns=[columnName])
    return df

def mapToNumber(df, valuesList, columnName):
    """
    Map ordinal column values to numbers based on their order in valuesList.
    """
    mapping = {value: idx for idx, value in enumerate(valuesList)}
    df[columnName] = df[columnName].map(mapping)
    return df

def dataEncoded(df):
    # Binary columns
    df = binaryConvert(df, 'Have_you_ever_had_suicidal_thoughts_')
    df = binaryConvert(df, 'Family_History_of_Mental_Illness')

    # Categorical columns
    Gender = ['Male', 'Female']
    Degree = ['STEM_Bach', 'Business', 'Arts', 'HighSchool', 'Graduate', 'PhD']

    df = oneHotEncoding(df, Gender, 'Gender')
    df = oneHotEncoding(df, Degree, 'Degree_Category')  # encode and drop Degree_Category automatically

    # Ordinal columns
    SleepDuration = ['<5','5-6','6-7','7-8','>8']
    DietaryHabits = ['unhealthy', 'moderate', 'healthy']

    df = mapToNumber(df, SleepDuration, 'Sleep_Duration')
    df = mapToNumber(df, DietaryHabits, 'Dietary_Habits')

    return df

def main():
    path = "data/cleaned/student_depression_cleaned.csv"
    df = pd.read_csv(path)
    df_encoded = dataEncoded(df)
    df_encoded.to_csv("data/encoded/student_depression_encoded.csv", index=False)

if __name__ == "__main__":
    main()
