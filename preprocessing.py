import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Loading bank-full dataset
file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\bank-full.csv"
data = pd.read_csv(file_path, sep=";")  # creates dataframe

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Label encode binary columns
binary_cols = ['default', 'housing', 'loan', 'y']
le = LabelEncoder()   # Transforms these binary columns to numerical data
for col in binary_cols:
    data[col] = le.fit_transform(data[col])

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'])
data = data.astype(int)

data = data.drop(columns=['duration'])

# Create the 'was_contacted' column [0 means not contacted, 1 means contacted]
data['was_contacted'] = (data['pdays'] != -1).astype(int)

# Step 2: Replace -1 in 'pdays' with NaN
data['pdays'] = data['pdays'].replace(-1, np.nan)

# Remove outlier from previous column manually
data = data[data['previous'] != 275]


# Save the preprocessed data
preprocessed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank-full\\bank_preprocessed1.csv"
data.to_csv(preprocessed_file_path, index=False)
print(f"Preprocessed data saved to {preprocessed_file_path}")
