import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------------
# Load the trained model
# -------------------------------
model_path = os.path.join("models", "xgboost_tuned.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("Student Depression Prediction")
st.write("Predict the likelihood of depression based on student features")

# -------------------------------
# User Inputs
# -------------------------------
Age = st.slider("Age", 15, 40, 20)

Have_suicidal_thoughts = st.selectbox(
    "Have you ever had suicidal thoughts?", ["No", "Yes"]
)
Academic_Pressure = st.slider("Academic Pressure", 0, 10, 5)
Financial_Stress = st.slider("Financial Stress", 0, 10, 5)
Family_History = st.selectbox(
    "Family History of Mental Illness", ["No", "Yes"]
)
CGPA = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
Work_Hours = st.slider("Work/Study Hours per Week", 0, 100, 40)
Sleep_Duration = st.slider("Average Sleep Duration (hrs)", 0, 24, 7)
Study_Satisfaction = st.slider("Study Satisfaction", 0, 10, 5)
Dietary_Habits = st.slider("Dietary Habits (1 bad - 5 good)", 1, 5, 3)

Gender = st.selectbox("Gender", ["Male", "Female"])
Degree = st.selectbox(
    "Degree Category",
    ["STEM_Bach", "Business", "Arts", "HighSchool", "Graduate", "PhD"]
)

# -------------------------------
# Encoding
# -------------------------------
def encode_yes_no(value):
    return 1 if value == "Yes" else 0

# Gender One-Hot
gender_data = {"Gender_Male": 0, "Gender_Female": 0}
gender_data[f"Gender_{Gender}"] = 1

# Degree One-Hot
degree_columns = [
    "Degree_Category_STEM_Bach",
    "Degree_Category_Business",
    "Degree_Category_Arts",
    "Degree_Category_HighSchool",
    "Degree_Category_Graduate",
    "Degree_Category_PhD"
]
degree_data = {col: 0 for col in degree_columns}
degree_data[f"Degree_Category_{Degree}"] = 1

# -------------------------------
# EXACT feature order expected by model
# -------------------------------
final_columns = [
    "Age",
    "Academic_Pressure",
    "CGPA",
    "Study_Satisfaction",
    "Sleep_Duration",
    "Dietary_Habits",
    "Have_you_ever_had_suicidal_thoughts_",
    "Work_Study_Hours",
    "Financial_Stress",
    "Family_History_of_Mental_Illness",
    "Gender_Male",
    "Gender_Female",
    "Degree_Category_STEM_Bach",
    "Degree_Category_Business",
    "Degree_Category_Arts",
    "Degree_Category_HighSchool",
    "Degree_Category_Graduate",
    "Degree_Category_PhD"
]

# -------------------------------
# Build DataFrame
# -------------------------------
input_dict = {
    "Age": Age,
    "Academic_Pressure": Academic_Pressure,
    "CGPA": CGPA,
    "Study_Satisfaction": Study_Satisfaction,
    "Sleep_Duration": Sleep_Duration,
    "Dietary_Habits": Dietary_Habits,
    "Have_you_ever_had_suicidal_thoughts_": encode_yes_no(Have_suicidal_thoughts),
    "Work_Study_Hours": Work_Hours,
    "Financial_Stress": Financial_Stress,
    "Family_History_of_Mental_Illness": encode_yes_no(Family_History),
    **gender_data,
    **degree_data
}

input_data = pd.DataFrame([input_dict])[final_columns]

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High likelihood of depression ({prob*100:.1f}%)")
    else:
        st.success(f"Low likelihood of depression ({(1-prob)*100:.1f}%)")

