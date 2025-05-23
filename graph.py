from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

unmodified_svm = {
    "Accuracy": 0.7022,
    "Precision": 0.5407,
    "Recall": 0.5709,
    "F1 Score": 0.5165
}

metrics_new = {
    "Accuracy": 0.7430,
    "Precision": 0.4838,
    "Recall": 0.5093,
    "F1 Score": 0.4894
}

df_unmodified_svm = pd.DataFrame(list(unmodified_svm.items()), columns=["Metrics", "Value"])

barplot = sns.barplot(x="Value", y="Metrics", data=df_unmodified_svm, palette='coolwarm');

for index, value in enumerate(df_unmodified_svm["Value"]):
    plt.text(value + 0.01, index, f'{value:.4f}', va='center');

plt.title("Unmodified SVM");
plt.grid(True, axis='x');
plt.xlabel("Score");
plt.xlim(0, 1);
plt.tight_layout();
plt.show();

df_metrics_new = pd.DataFrame(list(metrics_new.items()), columns=["Metrics", "Value"])

barplot = sns.barplot(x="Value", y="Metrics", data=df_metrics_new, palette='coolwarm');

for index, value in enumerate(df_metrics_new["Value"]):
    plt.text(value + 0.01, index, f'{value:.4f}', va='center');

plt.title("Modified SVM");
plt.grid(True, axis='x');
plt.xlabel("Score");
plt.xlim(0, 1);
plt.tight_layout();
plt.show();