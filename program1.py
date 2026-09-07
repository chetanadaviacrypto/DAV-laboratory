import numpy as np
import pandas as pd
np.random.seed(42)  
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': np.random.randint(20, 40, size=5),
    'Salary': np.random.randint(30000, 80000, size=5),
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR']
}


df = pd.DataFrame(data)

print(" Dataset ")
print(df, "\n")

print(" Summary Statistics ")
print(df.describe(), "\n")

print(" Group by Department:")
print(df.groupby('Department')['Salary'].mean(), "\n")

print("Rows in the dataset")
print(df.head())

print("display all column name:")
print(df.info(),"\n")

print(df.value_counts(),"\n")
