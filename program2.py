import numpy as np
import pandas as pd
# 1. Load CSV
df = pd.read_csv("data.csv")

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== LAST 5 ROWS ===")
print(df.tail())

# 2. Dataset structure
print("\n=== SHAPE (ROWS, COLUMNS) ===")
print(df.shape)

print("\n=== COLUMN NAMES ===")
print(df.columns)

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== DATASET INFORMATION ===")
df.info()

print("\n=== STATISTICAL SUMMARY ===")
print(df.describe())

# 3. Statistical operations
age = df["Age"]

print("\n=== AGE STATISTICS ===")
print("Mean:", age.mean())
print("Median:", age.median())
print("Mode:", age.mode().tolist())
print("Variance:", age.var())
print("Standard deviation:", age.std())
print("Minimum:", age.min())
print("Maximum:", age.max())
print("Count:", age.count())

# 4. Categorical operations
print("\n=== UNIQUE CITIES ===")
print(df["City"].unique())

print("\n=== NUMBER OF UNIQUE CITIES ===")
print(df["City"].nunique())

print("\n=== CITY FREQUENCY ===")
print(df["City"].value_counts())

# 5. Missing-value operations
print("\n=== MISSING VALUE MASK ===")
print(df.isnull())

print("\n=== MISSING VALUE COUNT ===")
print(df.isnull().sum())

print("\n=== MISSING VALUE PERCENTAGE ===")
print(df.isnull().mean() * 100)

print("\n=== DATA AFTER DROPPING MISSING ROWS ===")
print(df.dropna())

print("\n=== DATA AFTER FILLING MISSING VALUES ===")
filled_df = df.copy()
filled_df["Age"] = filled_df["Age"].fillna(filled_df["Age"].median())
filled_df["Salary"] = filled_df["Salary"].fillna(filled_df["Salary"].mean())
print(filled_df)

# 6. Duplicate operations
print("\n=== DUPLICATE ROW MASK ===")
print(df.duplicated())

print("\n=== DUPLICATE COUNT ===")
print(df.duplicated().sum())

print("\n=== DATA AFTER REMOVING DUPLICATES ===")
print(df.drop_duplicates())

# 7. Selection and filtering
print("\n=== SELECT ONE COLUMN ===")
print(df["Age"])

print("\n=== SELECT MULTIPLE COLUMNS ===")
print(df[["Age", "Salary"]])

print("\n=== SELECT FIRST 5 ROWS BY POSITION ===")
print(df.iloc[0:5])

print("\n=== CONDITIONAL SELECTION: AGE > 30 ===")
print(df[df["Age"] > 30])

# 8. Sort and rename
print("\n=== SORT BY AGE ===")
print(df.sort_values("Age"))

print("\n=== RENAME AGE COLUMN ===")
renamed_df = df.rename(columns={"Age": "Age_Years"})
print(renamed_df.columns)

# 9. Feature creation
print("\n=== CREATE BONUS COLUMN ===")
feature_df = df.copy()
feature_df["Bonus"] = feature_df["Salary"] * 0.10
print(feature_df)

print("\n=== DELETE BONUS COLUMN ===")
feature_df = feature_df.drop("Bonus", axis=1)
print(feature_df)

# 10. Group operations
print("\n=== GROUP BY DEPARTMENT ===")
print(df.groupby("Department").size())

print("\n=== AVERAGE SALARY BY DEPARTMENT ===")
print(df.groupby("Department")["Salary"].mean())

# 11. NumPy example
print("\n=== NUMPY EXAMPLE ===")
salary_values = df["Salary"].dropna().to_numpy()
print("Salary array:", salary_values)
print("NumPy mean salary:", np.mean(salary_values))

# 12. Concatenate and merge examples
df1 = df[["ID", "Name"]] if "Name" in df.columns else df[["ID", "City"]]
df2 = pd.DataFrame({
    "ID": [109, 110],
    "City": ["Mangaluru", "Dharwad"],
    "Salary": [43000, 48000]
})

print("\n=== CONCATENATE DATASETS ===")
print(pd.concat([df, df2], ignore_index=True))

department_info = pd.DataFrame({
    "Department": ["IT", "HR", "Finance"],
    "Manager": ["Manager A", "Manager B", "Manager C"]
})

print("\n=== MERGE DATASETS ===")
print(pd.merge(df, department_info, on="Department", how="left"))

print("\n=== EDA DEMONSTRATION COMPLETE ===")
