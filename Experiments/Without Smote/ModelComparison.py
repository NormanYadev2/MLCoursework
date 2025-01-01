# Compare results of Random Forest and Neural Network models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from prettytable import PrettyTable
from sklearn.metrics import classification_report

from prettytable import PrettyTable

from WithoutSmote import accuracy_rf_train, accuracy_rf_test, roc_auc_rf_test, roc_auc_nn_test, accuracy_nn_test, \
    accuracy_nn_train, y_pred_nn_test, y_pred_rf_test, y_test, y_pred_rf_test_prob, y_pred_nn_test_prob

# Create a table to display metrics
table = PrettyTable()
table.field_names = ["Metric", "Random Forest", "Neural Network"]

# Add rows to the table
table.add_row(["Training Accuracy", f"{accuracy_rf_train * 100:.2f}%", f"{accuracy_nn_train * 100:.2f}%"])
table.add_row(["Test Accuracy", f"{accuracy_rf_test * 100:.2f}%", f"{accuracy_nn_test * 100:.2f}%"])
table.add_row(["Test ROC AUC", f"{roc_auc_rf_test:.2f}", f"{roc_auc_nn_test:.2f}"])

# Display the table
print("\n--- Model Comparison Table ---")
print(table)




# Generate classification reports for both models
rf_report = classification_report(y_test, y_pred_rf_test, zero_division=0, output_dict=True)
nn_report = classification_report(y_test, y_pred_nn_test, zero_division=0, output_dict=True)

# Create a second table to compare precision, recall, and F1-score
table_2 = PrettyTable()
table_2.field_names = ["Metric", "Class 0 (RF)", "Class 0 (NN)", "Class 1 (RF)", "Class 1 (NN)"]

# Add rows for precision, recall, and F1-score
table_2.add_row([
    "Precision",
    f"{rf_report['0']['precision']:.2f}",
    f"{nn_report['0']['precision']:.2f}",
    f"{rf_report['1']['precision']:.2f}",
    f"{nn_report['1']['precision']:.2f}"
])
table_2.add_row([
    "Recall",
    f"{rf_report['0']['recall']:.2f}",
    f"{nn_report['0']['recall']:.2f}",
    f"{rf_report['1']['recall']:.2f}",
    f"{nn_report['1']['recall']:.2f}"
])
table_2.add_row([
    "F1-Score",
    f"{rf_report['0']['f1-score']:.2f}",
    f"{nn_report['0']['f1-score']:.2f}",
    f"{rf_report['1']['f1-score']:.2f}",
    f"{nn_report['1']['f1-score']:.2f}"
])

# Display the second table
print("\n--- Precision, Recall, F1-Score Comparison Table ---")
print(table_2)


# Plot ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_test_prob)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf_test:.2f})")

# Plot ROC Curve for Neural Network
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_nn_test_prob)
plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {roc_auc_nn_test:.2f})")

# Add plot details
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")  # Random guess line
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()