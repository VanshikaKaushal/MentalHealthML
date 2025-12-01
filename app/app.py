import streamlit as st
import pandas as pd
import pickle
import os


st.markdown("""
<style>
/* ===== Make ALL widget labels darker ===== */
.stSelectbox label,
.stNumberInput label,
.stSlider label,
.stMultiSelect label,
.stTextInput label,
.stRadio label,
.stCheckbox label,
.stDateInput label,
.stColorPicker label,
.stTimeInput label,
label {
    color: #1a1a1a !important;   /* dark charcoal */
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* ===== Sidebar labels ===== */
section[data-testid="stSidebar"] label {
    color: #1a1a1a !important;
}

/* ===== Sidebar title ("Input Student Information") ===== */
/* This targets st.sidebar.title() */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1a1a1a !important;
    font-weight: 700 !important;
}

/* Optional: make font slightly cleaner */
section[data-testid="stSidebar"] h2 {
    font-size: 1.3rem !important;
}
</style>
""", unsafe_allow_html=True)



# ------------------------------------
# Load Model
# ------------------------------------
model_path = os.path.join("models", "xgboost_tuned.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ------------------------------------
# Page Setup
# ------------------------------------
st.set_page_config(
    page_title="Depression Risk Predictor",
    layout="centered",
)

# ------------------------------------
# Custom CSS for background and text
# ------------------------------------
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #f0f8ff;  
        color: #000000;
    }

    /* Sidebar background */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #e6f2ff !important; 
    }
    
    /* Sidebar width */
    [data-testid="stSidebar"] {
        min-width: 340px !important;
        max-width: 340px !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-left: 20px;
        padding-right: 20px;
    }

    /* Customize button */
    .stButton>button {
        background-color: #4da6ff;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)




st.title("Student Depression Risk Prediction")
st.write(
    "Use this tool to estimate the likelihood of depression based on academic, lifestyle, and personal factors."
)

st.info(
    "This tool is **not a medical diagnosis**. It is a machine-learning model intended for educational use only."
)

# ------------------------------------
# Sidebar Inputs
# ------------------------------------
st.sidebar.header("Input Student Information")

Age = st.sidebar.slider("Age", 15, 40, 20)

Have_suicidal_thoughts = st.sidebar.selectbox(
    "Have you ever had suicidal thoughts?", ["No", "Yes"]
)

Academic_Pressure = st.sidebar.slider("Academic Pressure", 0, 10, 5)
Financial_Stress = st.sidebar.slider("Financial Stress", 0, 10, 5)

Family_History = st.sidebar.selectbox(
    "Family History of Mental Illness", ["No", "Yes"]
)

CGPA = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
Work_Hours = st.sidebar.slider("Work/Study Hours per Week", 0, 100, 40)
Sleep_Duration = st.sidebar.slider("Average Sleep Duration (hrs)", 0, 24, 7)
Study_Satisfaction = st.sidebar.slider("Study Satisfaction", 0, 10, 5)
Dietary_Habits = st.sidebar.slider("Dietary Habits (1 bad - 5 good)", 1, 5, 3)

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Degree = st.sidebar.selectbox(
    "Degree Category",
    ["STEM_Bach", "Business", "Arts", "HighSchool", "Graduate", "PhD"]
)

# ------------------------------------
# Encoding Functions
# ------------------------------------
def encode_yes_no(value):
    return 1 if value == "Yes" else 0

# Gender one-hot
gender_data = {"Gender_Male": 0, "Gender_Female": 0}
gender_data[f"Gender_{Gender}"] = 1

# Degree one-hot
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

# ------------------------------------
# Feature Order
# ------------------------------------
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

# ------------------------------------
# Build Model Input
# ------------------------------------
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

# ------------------------------------
# Prediction
# ------------------------------------
st.subheader("Prediction Result")

if st.button("Run Prediction"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    prob_percent = prob * 100

    # ---- Risk Level Visualization ----
    st.progress(float(prob))

    # ---- Text Output ----
    if prediction == 1:
        st.error(f"**High likelihood of depression: {prob_percent:.1f}%**")
    else:
        st.success(f"**Low likelihood of depression: {prob_percent:.1f}%**")

    # ---- Summary box ----
    st.subheader("Prediction Summary")
    st.write(
    f"""
    - **Probability of Depression:** {prob_percent:.1f}%  
    - **Probability of No Depression:** {(100 - prob_percent):.1f}%  
    - **Model Used:** XGBoost (Tuned)  
    """
    )

    # ---- Helpful Resources ----
    st.markdown("---")
    st.markdown("### Helpful Mental Health Resources")
    st.markdown(
    """
    - **988 Suicide & Crisis Lifeline (US)**  
    - **National Alliance on Mental Illness (NAMI):** https://nami.org  
    - **Mental health support at your school** (counseling center)  
    """
    )

