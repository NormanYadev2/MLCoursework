from prettytable import PrettyTable

# Create a PrettyTable object
table = PrettyTable()

# Define the field names (column headers)
table.field_names = ["Experiment", "Model", "Training Accuracy", "Test Accuracy", "Test ROC AUC"]

# Add rows to the table for each experiment
table.add_row(["Experiment 1 (No SMOTE, No Tuning)", "Random Forest", "99.85%", "88.17%", "0.79"])
table.add_row(["Experiment 1 (No SMOTE, No Tuning)", "Neural Network", "89.28%", "88.71%", "0.79"])
table.add_row(["Experiment 2 (No SMOTE, Tuning)", "Random Forest", "93.42%", "88.81%", "0.79"])
table.add_row(["Experiment 2 (No SMOTE, Tuning)", "Neural Network", "89.09%", "88.47%", "0.79"])
table.add_row(["Experiment 3 (SMOTE, No Tuning)", "Random Forest", "100.00%", "93.48%", "0.97"])
table.add_row(["Experiment 3 (SMOTE, No Tuning)", "Neural Network", "94.36%", "93.64%", "0.97"])
table.add_row(["Experiment 4 (SMOTE, Tuning)", "Random Forest", "95.17%", "91.25%", "0.97"])
table.add_row(["Experiment 4 (SMOTE, Tuning)", "Neural Network", "94.08%", "93.58%", "0.97"])

# Print the table
print(table)
