import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV


# Loading the preprocessed data
preprocessed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank-full\\bank_preprocessed1.csv"
data = pd.read_csv(preprocessed_file_path)

# change NaN values to mean for normalization
data['pdays'] = data['pdays'].fillna(data['pdays'].mean())


numeric_cols = ['age', 'pdays', 'previous', 'balance', 'campaign', 'day']

# Scaling numeric features
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Applying PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(data[numeric_cols])

# Create a new DataFrame for PCA-transformed features
pca_columns = [f'PCA_{i+1}' for i in range(X_pca.shape[1])]
pca_data = pd.DataFrame(X_pca, columns=pca_columns)

# Combining PCA data with other columns except numeric_cols
final_data = pd.concat([pca_data, data.drop(columns=numeric_cols)], axis=1)

# Handling missing values in the target variable
final_data['y'] = final_data['y'].fillna(final_data['y'].mode()[0])  # Fill NaNs with the mode

# Save the finalized dataset
finalized_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank-full\\bank_finalized1.csv"
final_data.to_csv(finalized_file_path, index=False)
print(f"Finalized data saved to {finalized_file_path}")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# Preparing data for modeling
X = final_data.drop(columns=['y'])
y = final_data['y']

# Split the data into train, validation, and test sets[70% for training and 15 % for each validation and test data]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# --- Hyperparameter Tuning for Random Forest ---
rf_model = RandomForestClassifier(random_state=42)

# Defining parameter grid for RandomizedSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', 0.1, 0.3, 0.5],
    'max_samples': [0.8, 0.9, 1.0],
    'bootstrap': [True],
    'oob_score': [True, False],
    'class_weight': ['balanced'],
    'warm_start': [True, False]

}

# Performing RandomizedSearchCV
rf_random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid_rf, n_iter=10, cv=3, random_state=42, verbose=2, n_jobs=-1)
rf_random_search.fit(X_train, y_train)

# Get the best model from RandomizedSearchCV
best_rf_model = rf_random_search.best_estimator_

# Predicting probabilities for the training and test sets
y_pred_rf_train_prob = best_rf_model.predict_proba(X_train)[:, 1]
y_pred_rf_test_prob = best_rf_model.predict_proba(X_test)[:, 1]

# Applying thresholding
threshold = 0.3  # Adjust this threshold based on desired precision/recall tradeoff
y_pred_rf_train = (y_pred_rf_train_prob >= threshold).astype(int)
y_pred_rf_test = (y_pred_rf_test_prob >= threshold).astype(int)

# Evaluating Random Forest (Training)
accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
print("\nRandom Forest Model Evaluation (Training):")
print(f"Training Accuracy: {accuracy_rf_train * 100:.2f}%")

# Evaluating Random Forest (Testing)
accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)
roc_auc_rf_test = roc_auc_score(y_test, y_pred_rf_test_prob)

print("\nRandom Forest Model Evaluation (Tuned with Threshold):")
print(f"Test Accuracy: {accuracy_rf_test * 100:.2f}%")
print(f"Test ROC AUC: {roc_auc_rf_test:.2f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_rf_test, zero_division=0))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_rf_test))

# --- Neural Network Model ---
# Building a neural network model using Keras
nn_model = Sequential()
nn_model.add(Input(shape=(X_train.shape[1],)))
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

# Compiling the model
nn_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Training the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

nn_model.fit(X_train, y_train, epochs=75, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Predicting the probabilities for the training and test sets
y_pred_nn_train_prob = nn_model.predict(X_train)
y_pred_nn_test_prob = nn_model.predict(X_test)

# Applying thresholding
y_pred_nn_train = (y_pred_nn_train_prob >= threshold).astype(int)
y_pred_nn_test = (y_pred_nn_test_prob >= threshold).astype(int)

# Evaluating Neural Network (Training)
accuracy_nn_train = accuracy_score(y_train, y_pred_nn_train)
print("\nNeural Network Model Evaluation (Training):")
print(f"Training Accuracy: {accuracy_nn_train * 100:.2f}%")

# Evaluating Neural Network (Testing)
accuracy_nn_test = accuracy_score(y_test, y_pred_nn_test)
roc_auc_nn_test = roc_auc_score(y_test, y_pred_nn_test_prob)

print("\nNeural Network Model Evaluation (Tuned with Threshold):")
print(f"Test Accuracy: {accuracy_nn_test * 100:.2f}%")
print(f"Test ROC AUC: {roc_auc_nn_test:.2f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_nn_test, zero_division=0))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_nn_test))


