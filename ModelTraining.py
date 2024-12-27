# Prepare data for modeling
X = final_data.drop(columns=['y'])
y = final_data['y']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Random Forest Model ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_rf * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# --- Neural Network Model ---
# Build a neural network model using Keras
nn_model = Sequential()
nn_model.add(Input(shape=(X_train.shape[1],)))  # Input layer
nn_model.add(Dense(64, activation='relu'))  # Hidden layer
nn_model.add(Dense(32, activation='relu'))  # Another hidden layer
nn_model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
nn_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluate the model
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)

# Evaluate Neural Network
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print("\nNeural Network Model Evaluation:")
print(f"Accuracy: {accuracy_nn * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))