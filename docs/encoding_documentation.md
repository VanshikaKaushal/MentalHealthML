# Data Encoding Module Documentation

**File:** `src/data_encoding.py`  
**Purpose:** Encode the cleaned student depression dataset so that all columns are fully numeric and ready for machine learning. Includes methods for binary conversion, one-hot encoding, and ordinal mapping.

---

## 1. `binaryConvert(df, columnName)`

**Description:**  
Converts binary columns with `'Yes'`/`'No'` values into numeric format (`1`/`0`).  

**Arguments:**  
- `df` (`pd.DataFrame`): Input DataFrame  
- `columnName` (`str`): Name of the column to convert  

**Returns:**  
- `pd.DataFrame`: DataFrame with the specified column converted to numeric values  

---

## 2. `oneHotEncoding(df, categories, columnName)`

**Description:**  
Performs one-hot encoding on a categorical column. Creates a new binary column for each category. The original column is dropped automatically.  

**Arguments:**  
- `df` (`pd.DataFrame`): Input DataFrame  
- `categories` (`list` of `str`): List of possible categories in the column  
- `columnName` (`str`): Column name to encode  

**Returns:**  
- `pd.DataFrame`: DataFrame with one-hot encoded columns  

---

## 3. `mapToNumber(df, valuesList, columnName)`

**Description:**  
Maps ordinal column values to integers based on their order in a provided list. First value in the list maps to `0`, second to `1`, and so on.  

**Arguments:**  
- `df` (`pd.DataFrame`): Input DataFrame  
- `valuesList` (`list` of `str`): Ordered list of possible values  
- `columnName` (`str`): Column name to map  

**Returns:**  
- `pd.DataFrame`: DataFrame with the specified column mapped to numeric values  

---

## 4. `dataEncoded(df)`

**Description:**  
Encodes the full dataset by applying `binaryConvert`, `oneHotEncoding`, and `mapToNumber` for all relevant columns. Automatically drops redundant columns like `Degree_Category` after one-hot encoding.  

**Processing Steps:**  
1. Converts binary columns:  
   - `Have_you_ever_had_suicidal_thoughts_`  
   - `Family_History_of_Mental_Illness`  
2. One-hot encodes categorical columns:  
   - `Gender` → `Gender_Male`, `Gender_Female`  
   - `Degree_Category` → `Degree_STEM_Bach`, `Degree_Business`, etc.  
3. Maps ordinal columns:  
   - `Sleep_Duration`  
   - `Dietary_Habits`  

**Returns:**  
- `pd.DataFrame`: Fully numeric DataFrame ready for machine learning  

---

## 5. `main()`

**Description:**  
Reads the cleaned CSV file, encodes the dataset using `dataEncoded()`, and saves the resulting fully numeric dataset to a new CSV file.  

**Input:**  
- `"data/cleaned/student_depression_cleaned.csv"`  

**Output:**  
- `"data/encoded/student_depression_encoded.csv"`  

---

**Notes:**  
- Ensure all categorical values in your CSV match the lists defined in the script (`Gender`, `Degree_Category`, `SleepDuration`, `DietaryHabits`).  
- The original `Degree_Category` column is automatically removed after encoding.  
- You can adjust the lists of categories or ordinal values if the dataset changes.  
