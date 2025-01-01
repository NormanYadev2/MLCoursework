# No smote and no tuning

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from FeatureEngineering import final_data, pca

# Save the finalized dataset
finalized_file_path = "C:\\Users\\Muralish\\Desktop\\Machine lerning coursework\\Data\\bank\\preprocessed_bankdata\\bank-full\\bank_finalized1.csv"
final_data.to_csv(finalized_file_path, index=False)
print(f"Finalized data saved to {finalized_file_path}")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# Preparing data for modeling
X = final_data.drop(columns=['y'])
y = final_data['y']

# Splitting the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# --- Random Forest Model (Without Hyperparameter Tuning) ---
rf_model = RandomForestClassifier(random_state=42)

# Fitting the Random Forest model
rf_model.fit(X_train, y_train)

# Predicting probabilities for the training and test sets
y_pred_rf_train_prob = rf_model.predict_proba(X_train)[:, 1]
y_pred_rf_test_prob = rf_model.predict_proba(X_test)[:, 1]

# Applying thresholding for class imbalance
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

print("\nRandom Forest Model Evaluation (Without Hyperparameter Tuning):")
print(f"Test Accuracy: {accuracy_rf_test * 100:.2f}%")
print(f"Test ROC AUC: {roc_auc_rf_test:.2f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_rf_test, zero_division=0))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_rf_test))

# --- Neural Network Model (Without Hyperparameter Tuning) ---
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

# Predicting probabilities for the training and test sets
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

print("\nNeural Network Model Evaluation (Without Hyperparameter Tuning):")
print(f"Test Accuracy: {accuracy_nn_test * 100:.2f}%")
print(f"Test ROC AUC: {roc_auc_nn_test:.2f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_nn_test, zero_division=0))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_nn_test))
