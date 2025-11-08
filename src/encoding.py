# This file encodes our cleaned data so that dataset is all fully numeric and ready for Machine Learning. 

import pandas as pd

def binaryConvert(df, columnName):

    df[columnName] = df[columnName].astype(str).replace({
        'Yes' : 1,
        'No' : 0
    })

    return df


def oneHotEncoding(df):

    return df

def dataEncoded(df, list):
    
    df = binaryConvert(df, 'Have_you_ever_had_suicidal_thoughts_' )
    df = binaryConvert(df, 'Family_History_of_Mental_Illness')

    # Categorical Columns
    Gender = ['Male', 'Female']
    Degree = ['STEM_Bach' , 'Business', 'Arts', 'HighSchool', 'Graduate', 'PhD']

    df = oneHotEncoding(df, Gender)
    df = oneHotEncoding(df, Degree)

    #Ordinal Columns 
    SleepDuration = ['<5','5-6','6-7','7-8','>8']

        
    return df


def main():
    path = "data/cleaned/student_depression_cleaned.csv"
    df = pd.read_csv(path)
    df_encoded = dataEncoded(df)

    df_encoded.to_csv("data/encoded/student_depression_encoded.csv", index=False)



if __name__ == "__main__":
    main()