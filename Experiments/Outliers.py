import pandas as pd

# Load preprocessed data
preprocessed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank-full\\bank_preprocessed1.csv"
final_data = pd.read_csv(preprocessed_file_path)

# List of numerical columns you want to check for outliers
numeric_cols = ['age', 'pdays', 'previous', 'balance', 'campaign', 'day']


# Function to detect outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


# Initialize counters for y=1 values
total_y_1 = final_data[final_data['y'] == 1].shape[0]
total_y_1_outliers = 0

# Iterate over numerical columns and detect outliers
for col in numeric_cols:
    outliers = detect_outliers_iqr(final_data, col)
    outliers_count = outliers.shape[0]
    y_1_outliers = outliers[outliers['y'] == 1].shape[0]

    # Add y=1 outliers to total count
    total_y_1_outliers += y_1_outliers

    # Print outliers count and the number of y=1 in outliers
    print(f"Column: {col}")
    print(f"Total Outliers: {outliers_count}")
    print(f"y=1 in Outliers: {y_1_outliers}")
    print("-" * 40)

# Print the total y=1 values and total y=1 values of outliers
print("\nTotal y=1 values in the dataset:", total_y_1)
print("Total y=1 values in outliers:", total_y_1_outliers)
