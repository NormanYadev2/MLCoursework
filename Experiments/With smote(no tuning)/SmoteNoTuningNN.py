# Neural Network Without Hyperparameter Tuning

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

# Loading the preprocessed data
preprocessed_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank-full\\bank_preprocessed1.csv"
data = pd.read_csv(preprocessed_file_path)


# Define numerical columns
numeric_cols = ['age', 'pdays', 'previous', 'balance', 'campaign', 'day']

# Split data into features (X) and target (y)
X = data.drop(columns=['y'])
y = data['y']

# Apply SMOTE for oversampling before scaling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale only numerical features
X_numerical = X_resampled[numeric_cols]
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_numerical_scaled)

# Create a new DataFrame for PCA-transformed features
pca_columns = [f'PCA_{i+1}' for i in range(X_pca.shape[1])]
pca_data = pd.DataFrame(X_pca, columns=pca_columns)

# Combine PCA data with other columns (except numerical columns)
X_non_numerical = X_resampled.drop(columns=numeric_cols)
final_data = pd.concat([pca_data, X_non_numerical], axis=1)

# Add the target variable 'y' back into the final_data
final_data['y'] = y_resampled

# Prepare data for modeling
X = final_data.drop(columns=['y'])
y = final_data['y']

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Neural Network Model
nn_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
nn_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
nn_model.fit(X_train, y_train, epochs=75, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Predict probabilities for the training and test sets
y_pred_nn_train_prob = nn_model.predict(X_train)
y_pred_nn_test_prob = nn_model.predict(X_test)

# Define a threshold for classification
threshold = 0.5
y_pred_nn_train = (y_pred_nn_train_prob >= threshold).astype(int)
y_pred_nn_test = (y_pred_nn_test_prob >= threshold).astype(int)

# Evaluate Neural Network (Training)
accuracy_nn_train = accuracy_score(y_train, y_pred_nn_train)
print("\nNeural Network Model Evaluation (Training):")
print(f"Training Accuracy: {accuracy_nn_train * 100:.2f}%")

# Evaluate Neural Network (Testing)
accuracy_nn_test = accuracy_score(y_test, y_pred_nn_test)
roc_auc_nn_test = roc_auc_score(y_test, y_pred_nn_test_prob)

print("\nNeural Network Model Evaluation:")
print(f"Test Accuracy: {accuracy_nn_test * 100:.2f}%")
print(f"Test ROC AUC: {roc_auc_nn_test:.2f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_nn_test, zero_division=0))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_nn_test))
