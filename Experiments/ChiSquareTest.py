import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset with the correct separator
preprocessed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\bank-full.csv"
data = pd.read_csv(preprocessed_file_path, sep=";")

# Clean column names by removing unwanted characters like double quotes and extra spaces
data.columns = data.columns.str.replace('"', '').str.strip()

# List of categorical columns
categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

# Perform chi-square tests for each categorical column
for col in categorical_columns:
    # Check if the column exists in the dataframe
    if col in data.columns:
        # Create a contingency table
        contingency_table = pd.crosstab(data[col], data['y'])

        # Perform the chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Print the result
        print(f"Chi-Square Test for {col}: p-value = {p}")
    else:
        print(f"Column '{col}' does not exist in the dataset.")
