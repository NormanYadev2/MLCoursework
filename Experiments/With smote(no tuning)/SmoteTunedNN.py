import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch  # Keras Tuner for hyperparameter tuning
from imblearn.over_sampling import SMOTE  # Import SMOTE

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
pca_columns = [f'PCA_{i + 1}' for i in range(X_pca.shape[1])]
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


# Define the model building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))

    # Add hidden layers with variable units
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                        activation='relu'))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model with variable learning rate
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0001, 0.01])),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


# Initialize Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=3,  # Number of hyperparameter combinations to try
    executions_per_trial=1,
    directory='tuner_results',
    project_name='bank_model_tuning'
)

# Run the search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32,
             callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# Retrieve the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best number of layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"Layer {i} units: {best_hps.get(f'units_{i}')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Train the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# Evaluate the model
y_pred_train_prob = best_model.predict(X_train)
y_pred_train = (y_pred_train_prob >= 0.5).astype(int)

y_pred_test_prob = best_model.predict(X_test)
y_pred_test = (y_pred_test_prob >= 0.5).astype(int)

# Calculate training accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_pred_test_prob)

# Print results
print("\nBest Model Evaluation:")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test ROC AUC: {roc_auc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))
