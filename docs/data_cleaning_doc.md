# Data Cleaning Module Documentation

**File:** `src/data_cleaning.py`  
**Purpose:** Clean and preprocess the student depression dataset. Includes methods for cleaning columns, categorical variables, and numeric features.

---

## 1. `load_data(path)`

**Description:**  
Loads the dataset from a CSV file and prints a preview including first 5 rows, shape, info, and missing values per column.  

**Arguments:**  
- `path` (str): Path to the CSV file. Default is `"data/raw/student_depression_dataset.csv"`  

**Returns:**  
- `pd.DataFrame`: Loaded dataset  

---

## 2. `clean_columns(df)`

**Description:**  
Cleans the DataFrame column names and removes unnecessary columns.  

**Steps:**  
1. Drops columns: `'id', 'Profession', 'Job Satisfaction', 'Work Pressure', 'City'`  
2. Standardizes column names: removes spaces, special characters, and replaces `'/'` with `'_'`.  

**Arguments:**  
- `df` (`pd.DataFrame`): Input dataset  

**Returns:**  
- `pd.DataFrame`: Dataset with cleaned columns  

---

## 3. `clean_sleep_duration(df)`

**Description:**  
Cleans the `Sleep_Duration` column.  

**Steps:**  
1. Removes quotes, converts text to lowercase, strips whitespace.  
2. Removes rows containing `"other"` or `"others"`.  
3. Standardizes ranges:  
   - `'less than 5 hours' → '<5'`  
   - `'5-6 hours' → '5-6'`  
   - `'6-7 hours' → '6-7'`  
   - `'7-8 hours' → '7-8'`  
   - `'more than 8 hours' → '>8'`  

**Arguments:**  
- `df` (`pd.DataFrame`): Input dataset  

**Returns:**  
- `pd.DataFrame`: Dataset with cleaned `Sleep_Duration`  

---

## 4. `clean_dietary_habits(df)`

**Description:**  
Cleans the `Dietary_Habits` column.  

**Steps:**  
1. Converts text to lowercase and strips whitespace.  
2. Removes rows with values `"other"` or `"others"`.  

**Arguments:**  
- `df` (`pd.DataFrame`): Input dataset  

**Returns:**  
- `pd.DataFrame`: Dataset with cleaned `Dietary_Habits`  

---

## 5. `clean_degree(df)`

**Description:**  
Cleans the `Degree` column.  

**Steps:**  
1. Converts text to lowercase and strips whitespace.  
2. Removes rows with values `"other"` or `"others"`.  

**Arguments:**  
- `df` (`pd.DataFrame`): Input dataset  

**Returns:**  
- `pd.DataFrame`: Dataset with cleaned `Degree`  

---

## 6. `create_DegreeCategory(df)`

**Description:**  
Creates a new column `Degree_Category` by mapping specific degrees to broader categories such as STEM, Arts, Business, Graduate, High School, or PhD.  

**Steps:**  
1. Standardizes text by converting to lowercase, stripping spaces, and removing quotes.  
2. Maps degree names to categories:  

   | Degree         | Category        |
   |----------------|----------------|
   | B.Arch, B.Tech, BCA, BE, BSc, B.Pharm | STEM_Bach |
   | MBBS | STEM |
   | B.Com, BBA | Business |
   | BA, B.Ed, BHM, LLB | Arts |
   | M.Com, M.Ed, M.Pharm, M.Tech, MA, MBA, MCA, MD, ME, MHM, MAC | Graduate |
   | PhD | PhD |
   | Class 12 | HighSchool |
   | Others / unmapped | Other |

**Arguments:**  
- `df` (`pd.DataFrame`): Input dataset  

**Returns:**  
- `pd.DataFrame`: Dataset with new `Degree_Category` column  

---

## 7. `clean_CGPA(df)`

**Description:**  
Cleans the `CGPA` column.  

**Steps:**  
1. Removes quotes and strips whitespace.  
2. Removes rows with non-numeric values.  
3. Converts values to float and rounds to nearest integer.  

**Arguments:**  
- `df` (`pd.DataFrame`): Input dataset  

**Returns:**  
- `pd.DataFrame`: Dataset with cleaned `CGPA`  

---

## 8. `clean_values(df)`

**Description:**  
Applies all cleaning functions to the dataset in sequence.  

**Steps:**  
1. `clean_sleep_duration(df)`  
2. `clean_CGPA(df)`  
3. `clean_dietary_habits(df)`  
4. `clean_degree(df)`  
5. `create_DegreeCategory(df)`  

**Arguments:**  
- `df` (`pd.DataFrame`): Input dataset  

**Returns:**  
- `pd.DataFrame`: Fully cleaned dataset ready for analysis or ML  

---

## 9. `main()`

**Description:**  
Runs the complete cleaning pipeline:  
- Loads raw dataset  
- Cleans columns and values  
- Saves the cleaned dataset to `"data/cleaned/student_depression_cleaned.csv"`  

**Arguments:**  
- None  

**Returns:**  
- None (saves cleaned CSV to disk)
