import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\bank-full.csv"
data = pd.read_csv(file_path, sep=";")  # creates dataframe

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Label encode binary columns
binary_cols = ['default', 'housing', 'loan', 'y']
le = LabelEncoder()   #Transforms these binary columns to numerical data
for col in binary_cols:
    data[col] = le.fit_transform(data[col])

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome']) #Convert category variable to binary columns
data = data.astype(int)

# Save the preprocessed data
preprocessed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\\\bank-full\\bank_preprocessed.csv"
data.to_csv(preprocessed_file_path, index=False)
print(f"Preprocessed data saved to {preprocessed_file_path}")

