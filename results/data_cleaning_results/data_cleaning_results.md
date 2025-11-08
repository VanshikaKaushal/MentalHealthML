# Data Cleaning Report – Student Mental Health Project

**Date**: 2025-11-08  
**Author**: Amanda Lan & Vanshika Kaushal  
**Script**: `src/data_cleaning.py`

---

## 1. Objective  
Load raw student mental health data, inspect it, clean missing values and standardize features in preparation for machine learning.  

---

## 2. Source Dataset  
- **Filename**: `data/raw/student_depression_dataset.csv`  
- **Description**: A dataset of student records with demographic, academic, lifestyle, and mental-health indicators.  
- **Number of records (raw)**: 27901 x 19 fields  
- **Fields**: Provide a table or list of the columns and their description.  
  | Field | Description |
  |-------|-------------|
  | id | A unique identifier assigned to each student record in the dataset.|
  | Gender | The gender of the student (e.g., Male, Female, Other). This helps in analyzing gender-specific trends in mental health. |
  | Age | The age of the student in years. |
  | City | The city or region where the student resides, providing geographical context for the analysis.  |
  | Profession | The field of work or study of the student, which may offer insights into occupational or academic stress factors. |
  | Academic Pressure | A measure indicating the level of pressure the student faces in academic settings. This could include stress from exams, assignments, and overall academic expectations. |
  | Work Pressure | A measure of the pressure related to work or job responsibilities, relevant for students who are employed alongside their studies. |
  | CGPA | The cumulative grade point average of the student, reflecting overall academic performance. |
  | Study Satisfaction | An indicator of how satisfied the student is with their studies, which can correlate with mental well-being. |
  | Job Satisfaction | A measure of the student’s satisfaction with their job or work environment, if applicable. |
  | Sleep Duration | The average number of hours the student sleeps per day, which is an important factor in mental health. |
  | Dietary Habits | An assessment of the student’s eating patterns and nutritional habits, potentially impacting overall health and mood. |
  | Degree | The academic degree or program that the student is pursuing. |
  | Have you ever had suicidal thoughts ? | A binary indicator (Yes/No) that reflects whether the student has ever experienced suicidal ideation. |
  | Work/Study Hours | The average number of hours per day the student dedicates to work or study, which can influence stress levels. |
  | Financial Stress | A measure of the stress experienced due to financial concerns, which may affect mental health. |
  | Family History of Mental Illness | Indicates whether there is a family history of mental illness (Yes/No), which can be a significant factor in mental health predispositions. |
  | Depression | The target variable that indicates whether the student is experiencing depression (Yes/No). This is the primary focus of the analysis. |
  | … | … |

---

## 3. Loading & Initial Inspection  
- Show the first few rows (head) of the dataset.  
- Report shape (rows × columns).  
- Report missing values per column.  
- Report duplicate count or other relevant diagnostics.

```text
Example output:
Shape: (800, 17)
Missing values:
Gender          0
Age             2
Sleep Duration  1
…              …
Duplicates removed: 5

```text
First 5 rows:
   id  Gender   Age  ... Financial Stress Family History of Mental Illness  Depression
0   2    Male  33.0  ...              1.0                               No           1
1   8  Female  24.0  ...              2.0                              Yes           0
2  26    Male  31.0  ...              1.0                              Yes           0
3  30  Female  28.0  ...              5.0                              Yes           1
4  32  Female  25.0  ...              1.0                               No           0

[5 rows x 18 columns]

Shape: (27901, 18)

Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 27901 entries, 0 to 27900
Data columns (total 18 columns):
 #   Column                                 Dtype  
---  ------                                 -----
 1   id                                     int64 
 2   Gender                                 object 
 3   Age                                    float64 
 4   City                                   object 
 5   Profession                             object 
 6   Academic Pressure                      float64 
 7   Work Pressure                          float64 
 8   CGPA                                   float64 
 9   Study Satisfaction                     float64 
 10  Job Satisfaction                       float64 
 11  Sleep Duration                         object 
 12  Dietary Habits                         object 
 13  Degree                                 object 
 14  Have you ever had suicidal thoughts ?  object 
 15  Work/Study Hours                       float64 
 16  Financial Stress                       object 
 17  Family History of Mental Illness       object 
 18  Depression                             int64 
dtypes: float64(7), int64(2), object(9)
memory usage: 3.8+ MB

Missing values per column:
id                                       0
Gender                                   0
Age                                      0
City                                     0
Profession                               0
Academic Pressure                        0
Work Pressure                            0
CGPA                                     0
Study Satisfaction                       0
Job Satisfaction                         0
Sleep Duration                           0
Dietary Habits                           0
Degree                                   0
Have you ever had suicidal thoughts ?    0
Work/Study Hours                         0
Financial Stress                         0
Family History of Mental Illness         0
Depression                               0
dtype: int64
```
