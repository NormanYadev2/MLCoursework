import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\bank.csv"
data = pd.read_csv(file_path, sep=";")

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Label encode binary columns
binary_cols = ['default', 'housing', 'loan', 'y']
le = LabelEncoder()
for col in binary_cols:
    data[col] = le.fit_transform(data[col])

# One-hot encode other categorical columns
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'], drop_first=True)

# Scale numeric features
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

processed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_data\\bank_preprocessed.csv"
data.to_csv(processed_file_path, index=False)
processed_file_path

# Split features and target
X = data.drop(columns=['y'])  # Independent variables
y = data['y']  # Target variable

# Split into train and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Then, split the remaining data into validation (50% of the remaining, i.e., 10% of the total) and test (10% of the
# total)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

