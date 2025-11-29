# Student Depression Prediction Using Machine Learning

Contributors : Amanda Lan and Vanshika Kaushal

## 1. Introduction

Depression among students has become a critical global concern, affecting academic performance, social functioning, and overall well-being. With increasing academic pressure, financial stress, competitive environments, and lifestyle imbalances, students are becoming more vulnerable to mental-health challenges. Early detection of depressive tendencies is essential because timely awareness can help students seek support and institutions design preventive interventions. However, many students do not openly discuss their mental-health struggles, making data-driven prediction methods a valuable supplementary tool.

This project focuses on building a machine-learning model capable of predicting whether a student is likely to experience depression based on academic, lifestyle, and demographic indicators. By learning patterns from real student responses, the model aims to identify risk factors and contribute to proactive mental-health support systems.

## Dataset Description

The dataset used in this project is the **Student Depression Dataset** from Kaggle. It contains data gathered through self-reported surveys. Each record includes **17 input features**, such as:

- **Demographic attributes:** Gender, Age, City  
- **Academic indicators:** Academic Pressure, CGPA, Study Satisfaction  
- **Lifestyle factors:** Sleep Duration, Dietary Habits, Work/Study Hours  
- **Psychological stressors:** Suicidal Thoughts, Financial Stress  
- **Family background:** History of mental illness  

The **target variable** is `Depression`, a binary label where:

- **0 — No depression symptoms reported**  
- **1 — Depression symptoms reported**

This diverse mix of numeric and categorical features offers a strong foundation for using machine learning to uncover meaningful mental-health patterns among students.

---

## 2. Data Understanding
### 2.1 Dataset Overview

The original dataset, the **Student Depression Dataset** from Kaggle, contains a large collection of student responses related to academic life, lifestyle habits, stress levels, and mental health indicators.

Before cleaning, the dataset consisted of:

- **Rows:** 27,000+  
- **Columns:** 18  
- **Target Variable:** `Depression` (0 = Not Depressed, 1 = Depressed)

This dataset provides substantial diversity across demographics, degree types, sleep habits, dietary behaviors, and psychological stress levels, making it well-suited for building a predictive model.

After applying our cleaning pipeline (`data_cleaning.py`), we removed unnecessary columns such as `id`, `City`, `Profession`, `Job Satisfaction`, and `Work Pressure`, standardized inconsistent formats (sleep duration, dietary habits, CGPA, degree names), and handled invalid entries. Work Pressure was removed because all students did not have a part time job.

Once the dataset was encoded using the full encoding pipeline (`data_encoding.py`), all features were converted into fully numeric form, enabling effective training of machine learning models.


### 2.2 Key Observations from EDA
- Important patterns from numeric features

Academic_Pressure — strong positive separation: depressed students report substantially higher pressure (median ≈ 3.69 vs 2.36). One of the strongest numeric predictors.

Study_Satisfaction — strong negative separation: depressed students report lower satisfaction (mean ≈ 2.75 vs 3.21); statistically highly significant and a top predictor.

Work_Study_Hours — depressed students spend more hours (median ≈ 8 vs 6); distribution shifted upward for the depressed group, indicating workload as a relevant factor.

CGPA — virtually identical distributions across groups; weak predictor despite being an academic outcome.

Age — dataset concentrated on 18–30; younger students are slightly more likely to be depressed, but age offers only modest predictive value.

- Significant categorical trends 

Have_you_ever_had_suicidal_thoughts_ — very strong differentiator: the majority of depressed cases report suicidal thoughts (very high overlap); behaves like a proxy for severe depression.

Financial_Stress — clear monotonic relationship: higher financial stress → substantially higher depression prevalence (level 5 heavily skewed to depressed).

Dietary_Habits — healthier diets associate with lower depression rates; unhealthy diet shows markedly higher depression prevalence.

Sleep_Duration — short sleep (<5 hours) associates with higher depression; relationship is non-linear and interacts with stress/workload.

Degree_Category — lower educational categories (HighSchool) and STEM_Bach show elevated depression counts; Graduate category is more balanced.

- Correlation heatmap findings  

Mental-health cluster: Depression, Anxiety, and Stress are tightly positively correlated.

Positive correlates with Depression: Academic_Pressure, Financial_Stress, Work_Study_Hours (moderate correlations).

Negative correlates with Depression: Study_Satisfaction and sleep-related quality/quantity show moderate negative correlations.

Weak correlations: CGPA shows little-to-no positive correlation with Depression and is negatively correlated with Academic_Pressure.

No contradictory correlations detected; patterns align with expected psychological relationships.

- Any surprising or important insights  

Academic performance (CGPA) is not a driver of depression here — subjective measures (pressure, satisfaction) matter far more than grades.

Suicidal ideation behaves as the largest categorical differentiator and should be treated carefully in modeling and ethical review (high-risk flag).

Multiple stress-related variables cluster together (pressure, financial stress, work hours, low sleep, low satisfaction), indicating modeling should prioritize these features and consider interaction terms.

Class imbalance: the target is imbalanced (≈58.6% depressed); use class-weighting, resampling, or appropriate metrics (F1/precision–recall) during model development.

Data quality is high: no missing values or duplicates, enabling direct modeling without imputation.

---

 
## 3. Data Cleaning & Preprocessing

### 3.1 Cleaning Steps
To prepare the dataset for modeling, the following cleaning tasks were performed:

- Dropped irrelevant columns such as `id`, `Profession`, `Job Satisfaction`, `Work Pressure`, and `City`.
- Standardized text fields by converting to lowercase, stripping spaces, and removing special characters.
- Cleaned specific columns:
  - **Degree**: normalized text and removed ambiguous values (“other/others”) before grouping into broader categories.
  - **Dietary_Habits**: standardized labels and removed invalid “other” entries.
  - **Sleep_Duration**: converted all variations into five standard ranges (`<5`, `5–6`, `6–7`, `7–8`, `>8`).
  - **CGPA**: cleaned invalid values, ensured numeric type, and applied rounding.
- Replaced missing numeric values with the mean of the corresponding column.

---

### 3.2 Encoding
Categorical features were transformed as follows:

- **Binary Encoding**:
  - `Have_you_ever_had_suicidal_thoughts_` → 0/1  
  - `Family_History_of_Mental_Illness` → 0/1  

- **One-Hot Encoding**:
  - `Gender` → Male, Female  
  - `Degree_Category` → STEM_Bach, Business, Arts, HighSchool, Graduate, PhD

- **Ordinal Encoding**:
  - `Sleep_Duration` → `<5`, `5–6`, `6–7`, `7–8`, `>8`
  - `Dietary_Habits` → unhealthy, moderate, healthy

This resulted in a fully numeric dataset ready for model training.

---

### 3.3 Train–Test Split
- Used an **80/20 split** for training and testing.
- Applied **stratification** on the target variable (`Depression`) to maintain class balance.
- Saved the outputs in:
  - `data/training/X_train.csv`
  - `data/training/y_train.csv`
  - `data/testing/X_test.csv`
  - `data/testing/y_test.csv`


---

## 4. Feature Engineering

In this project, only light feature engineering was performed, focused mainly on improving the interpretability of the dataset.

### 4.1 New Features
- **Degree_Category**:  
  The original `Degree` column contained dozens of specific degree names. 
  These were grouped into broader, meaningful categories such as:
  - STEM_Bach  
  - Business  
  - Arts  
  - HighSchool  
  - Graduate  
  - PhD  

  This reduced noise in the model and allowed for more stable one-hot encoding.

### 4.2 Transformations
- Standardized categorical text values (lowercasing, trimming, removing special characters).
- Converted ordinal categories (`Sleep_Duration`, `Dietary_Habits`) into numeric scales.

### 4.3 Removed Features
Some features were removed because they were irrelevant, noisy, or not useful for prediction:
- `id`
- `Profession`
- `Job Satisfaction`
- `Work Pressure`
- `City`

These removals simplified the dataset and reduced unnecessary dimensionality.

Overall, feature engineering in this project was intentionally minimal, focusing on clarity rather than adding synthetic complexity.
 

---

 
## 5. Modelling Approach

### 5.1 Models Used
To evaluate a range of linear and non-linear classifiers, the following models were implemented:

- **Logistic Regression** – simple and interpretable baseline model  
- **Random Forest** – non-linear ensemble model capable of capturing complex patterns  
- **XGBoost** – high-performance gradient boosting model for tabular data  
- **Tuned XGBoost (Optuna)** – optimized version with automated hyperparameter search  

### 5.2 Model Performance Summary
| Model | Accuracy | Notes |
|-------|----------|-------|
| **Logistic Regression** | **0.8520** | Strong baseline performance. |
| **Random Forest** | **0.8412** | Slightly lower accuracy; less stable. |
| **XGBoost** | **0.8521** | Strong accuracy with better recall. |
| **Tuned XGBoost (Optuna)** | **0.8545** | **Best-performing model; chosen as final.** |

### 5.3 Hyperparameter Tuning (XGBoost)
Hyperparameter optimization was performed using **Optuna** with 30 trials.

- **Optimization Method:** Optuna TPE Sampler  
- **Objective:** Maximize validation accuracy  
- **Parameters Tuned:**  
  - `n_estimators`  
  - `max_depth`  
  - `learning_rate`  
  - `subsample`  
  - `colsample_bytree`  
  - `min_child_weight`  
  - `gamma`

**Best Parameters Found:**
```json
{
  "n_estimators": 418,
  "max_depth": 3,
  "learning_rate": 0.0548,
  "subsample": 0.9272,
  "colsample_bytree": 0.7776,
  "min_child_weight": 9,
  "gamma": 1.4255
}
```
---
 
## 6. Final Model Selection

**Selected Model:** Tuned XGBoost  

**Justification:**  
- Achieved the **highest accuracy** among all models: 0.8545.  
- Showed **balanced performance** for both classes: high recall for class 1 (only 374 misclassifications) and stable predictions for class 0 (1872 correct predictions).  
- Outperformed Logistic Regression and Random Forest in terms of overall F1-score and stability across metrics.  
- Hyperparameter tuning further improved performance slightly over the baseline XGBoost.

**Strengths:**  
- Handles non-linear relationships and interactions between features effectively.  
- Robust to multicollinearity and less sensitive to outliers.  
- Can capture feature importance, allowing interpretation of key predictors like `Have_you_ever_had_suicidal_thoughts_`, `Academic_Pressure`, and `Financial_Stress`.

**Weaknesses:**  
- Slightly more complex and slower to train compared to Logistic Regression.  
- Requires careful hyperparameter tuning to avoid overfitting.  
- Interpretability is lower than simple linear models.

---
## 7. Discussion / Errors / Limitations
- **Possible dataset biases:**  
  - Self-reported survey responses may introduce bias.  
  - Certain demographic groups may be over- or under-represented.  
- **Class imbalance issues:**  
  - The dataset has slightly more non-depressed than depressed cases, but stratified splitting mitigated most imbalance issues.  
- **Limitations of features:**  
  - Important factors like social support, therapy history, or genetic predisposition are not included.  
  - Some features (e.g., CGPA, Work_Study_Hours) are coarse indicators and may not capture nuances.  
- **Limitations of models:**  
  - Logistic Regression assumes linear relationships and may miss complex interactions.  
  - Random Forest may overfit to noise in some features.  
  - XGBoost, while accurate, may be harder to interpret for non-technical users.  

## 8. Ethical Considerations
- **Sensitivity of mental health predictions:**  
  - Predictions could impact individuals’ mental health if misused.  
- **Bias & fairness concerns:**  
  - Model may inherit biases from survey responses or feature representation.  
- **Responsible usage of predictions:**  
  - Model outputs should be used as guidance only, not as clinical diagnosis.  
  - Proper disclaimers and mental health support should accompany any deployment.  

## 9. Conclusion
- **Summary of findings:**  
  - XGBoost (tuned) provided the best balance of accuracy and recall for detecting depression risk.  
  - Key contributing factors: suicidal thoughts, academic pressure, financial stress, and dietary habits.  
- **Real-life usage:**  
  - Could be used as an early screening tool for student mental health interventions.  
- **Future improvements:**  
  - Include additional psychosocial features (e.g., social support, therapy history).  
  - Explore deep learning models for richer feature interactions.  
  - Perform SHAP or other interpretability analyses for actionable insights.

---

## 12. References
- [Dataset source  ](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- [IBM Machine Learning](https://www.ibm.com/think/topics/machine-learning) 
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) 
- [numpy](https://numpy.org)
- [xgBoost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)
- [skilit](https://scikit-learn.org/stable/)
- [pickle](https://docs.python.org/3/library/pickle.html)

