from prettytable import PrettyTable

# Results for Random Forest
rf_results = {
    "Training Accuracy": 95.17,
    "Testing Accuracy": 91.25,
    "Test ROC AUC": 0.97
}
rf_classification_report = {
    "precision": [0.93, 0.89],
    "recall": [0.89, 0.94],
    "f1-score": [0.91, 0.91]
}
rf_confusion_matrix = [[10628, 1349], [746, 11230]]

# Results for Neural Network
nn_results = {
    "Training Accuracy": 94.08,
    "Testing Accuracy": 93.58,
    "Test ROC AUC": 0.97
}
nn_classification_report = {
    "precision": [0.90, 0.98],
    "recall": [0.98, 0.89],
    "f1-score": [0.94, 0.93]
}
nn_confusion_matrix = [[7806,179], [846, 7138]]

# Table 1: Accuracy and ROC AUC
accuracy_table = PrettyTable()
accuracy_table.field_names = ["Model", "Training Accuracy (%)", "Testing Accuracy (%)", "Test ROC AUC"]
accuracy_table.add_row(["Random Forest", rf_results["Training Accuracy"], rf_results["Testing Accuracy"], rf_results["Test ROC AUC"]])
accuracy_table.add_row(["Neural Network", nn_results["Training Accuracy"], nn_results["Testing Accuracy"], nn_results["Test ROC AUC"]])

# Table 2: Precision, Recall, F1-Score
classification_table = PrettyTable()
classification_table.field_names = ["Metric", "Random Forest", "Neural Network"]
classification_table.add_row(["Precision (Class 0)", rf_classification_report["precision"][0], nn_classification_report["precision"][0]])
classification_table.add_row(["Precision (Class 1)", rf_classification_report["precision"][1], nn_classification_report["precision"][1]])
classification_table.add_row(["Recall (Class 0)", rf_classification_report["recall"][0], nn_classification_report["recall"][0]])
classification_table.add_row(["Recall (Class 1)", rf_classification_report["recall"][1], nn_classification_report["recall"][1]])
classification_table.add_row(["F1-Score (Class 0)", rf_classification_report["f1-score"][0], nn_classification_report["f1-score"][0]])
classification_table.add_row(["F1-Score (Class 1)", rf_classification_report["f1-score"][1], nn_classification_report["f1-score"][1]])

# Print Results
print("\nTable 1: Accuracy and ROC AUC Comparison")
print(accuracy_table)

print("\nTable 2: Precision, Recall, F1-Score Comparison")
print(classification_table)

# Confusion Matrices
print("\nRandom Forest Confusion Matrix:")
rf_cm_table = PrettyTable()
rf_cm_table.field_names = ["", "Predicted 0", "Predicted 1"]
rf_cm_table.add_row(["Actual 0", rf_confusion_matrix[0][0], rf_confusion_matrix[0][1]])
rf_cm_table.add_row(["Actual 1", rf_confusion_matrix[1][0], rf_confusion_matrix[1][1]])
print(rf_cm_table)

print("\nNeural Network Confusion Matrix:")
nn_cm_table = PrettyTable()
nn_cm_table.field_names = ["", "Predicted 0", "Predicted 1"]
nn_cm_table.add_row(["Actual 0", nn_confusion_matrix[0][0], nn_confusion_matrix[0][1]])
nn_cm_table.add_row(["Actual 1", nn_confusion_matrix[1][0], nn_confusion_matrix[1][1]])
print(nn_cm_table)
