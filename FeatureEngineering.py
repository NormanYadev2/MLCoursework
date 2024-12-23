import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the preprocessed data
preprocessed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank_preprocessed.csv"
data = pd.read_csv(preprocessed_file_path)

# Feature Engineering
data['balance_per_campaign'] = data['balance'] / data['campaign']
data['duration_per_day'] = data['duration'] / data['day']


# Log transformations (replace negative and zero values with a small positive constant before applying log1p)
#Handling Skewed Data
data['log_balance'] = np.log1p(data['balance'].clip(lower=1e-6))  #np.log1p - computes ln(1+x)
data['log_duration'] = np.log1p(data['duration'].clip(lower=1e-6))


# Scale numeric features
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
                'balance_per_campaign', 'duration_per_day', 'log_balance', 'log_duration']

# Replace infinities and NaN before scaling
data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
data[numeric_cols] = data[numeric_cols].fillna(0)  # Replace NaNs with 0

scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols]) #normalize the data so mean will be 0 and sd will be 1

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(data[numeric_cols])

# Create a new DataFrame for PCA-transformed features
pca_columns = [f'PCA_{i+1}' for i in range(X_pca.shape[1])]
pca_data = pd.DataFrame(X_pca, columns=pca_columns)

# Append the target variable to the PCA-transformed DataFrame
pca_data['y'] = data['y'].values

# Save the finalized dataset
finalized_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank_finalized.csv"
pca_data.to_csv(finalized_file_path, index=False)
print(f"Finalized data saved to {finalized_file_path}")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
