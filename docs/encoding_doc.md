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
Maps ordinal column values to integers based on their order in a provided list. First value in the list maps to `0`, seco
