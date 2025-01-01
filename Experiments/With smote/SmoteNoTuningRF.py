# RF model without parameter tuning using SMOTE

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Loading the preprocessed data
preprocessed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank-full\\bank_preprocessed1.csv"
data = pd.read_csv(preprocessed_file_path)


# Define numerical columns
numeric_cols = ['age', 'pdays', 'previous', 'balance', 'campaign', 'day']

# Split data into features (X) and target (y)
X = data.drop(columns=['y'])
y = data['y']

# --- Apply SMOTE for oversampling before scaling ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# --- Scale only numerical features ---
# Extract numerical columns for scaling
X_numerical = X_resampled[numeric_cols]

# Scale the numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_numerical_scaled)

# Create a new DataFrame for PCA-transformed features
pca_columns = [f'PCA_{i+1}' for i in range(X_pca.shape[1])]
pca_data = pd.DataFrame(X_pca, columns=pca_columns)

# --- Combine PCA data with other columns (except numerical columns) ---
# Keep non-numeric features (categorical and others)
X_non_numerical = X_resampled.drop(columns=numeric_cols)

# Concatenate PCA data with non-numerical features
final_data = pd.concat([pca_data, X_non_numerical], axis=1)

# Add the target variable 'y' back into the final_data
final_data['y'] = y_resampled

# Handle missing values in the target variable
final_data['y'] = final_data['y'].fillna(final_data['y'].mode()[0])  # Fill NaNs with the mode

# Save the finalized dataset
finalized_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank-full\\bank_finalized1.csv"
final_data.to_csv(finalized_file_path, index=False)
print(f"Finalized data saved to {finalized_file_path}")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# Prepare data for modeling
X = final_data.drop(columns=['y'])
y = final_data['y']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_resampled, test_size=0.3, random_state=42, stratify=y)

# --- Random Forest Model ---
rf_model = RandomForestClassifier(random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Predict probabilities for the training and test sets
y_pred_rf_train_prob = rf_model.predict_proba(X_train)[:, 1]
y_pred_rf_test_prob = rf_model.predict_proba(X_test)[:, 1]

# Convert probabilities to class labels
y_pred_rf_train = (y_pred_rf_train_prob >= 0.5).astype(int)
y_pred_rf_test = (y_pred_rf_test_prob >= 0.5).astype(int)

# Evaluate Random Forest (Training)
accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
print("\nRandom Forest Model Evaluation (Training):")
print(f"Training Accuracy: {accuracy_rf_train * 100:.2f}%")

# Evaluate Random Forest (Testing)
accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)
roc_auc_rf_test = roc_auc_score(y_test, y_pred_rf_test_prob)

print("\nRandom Forest Model Evaluation (Test):")
print(f"Test Accuracy: {accuracy_rf_test * 100:.2f}%")
print(f"Test ROC AUC: {roc_auc_rf_test:.2f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_rf_test, zero_division=0))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_rf_test))
