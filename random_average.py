import logging
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from kernerls import Linear, RBF
from SVM_model import SVM

logging.basicConfig(level=logging.DEBUG)

def compute_metrics(y_true, y_pred, pos_label=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_pred == pos_label) & (y_true == pos_label))
    TN = np.sum((y_pred != pos_label) & (y_true != pos_label))
    FP = np.sum((y_pred == pos_label) & (y_true != pos_label))
    FN = np.sum((y_pred != pos_label) & (y_true == pos_label))
    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

def generate_and_evaluate(n_datasets=10):
    results = {"Linear": [], "RBF": []}

    for i in range(n_datasets):
        X, y = make_classification(
            n_samples=1200, n_features=10, n_informative=5,
            n_classes=2, class_sep=1.75, random_state=1111 + i
        )
        y = (y * 2) - 1  # Convert y to {-1, 1}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

        for kernel_name, kernel in [("Linear", Linear())]:
            model = SVM(max_iter=500, kernel=kernel, C=0.6)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = compute_metrics(y_test, predictions, pos_label=1)
            results[kernel_name].append(metrics)
            print(f"Dataset {i+1} - Kernel {kernel_name} - Accuracy: {metrics['accuracy']:.4f}")

    for kernel_name, metrics_list in results.items():
        print(f"\nAverage Metrics for {kernel_name} Kernel With randomly generated Dataset:")
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    generate_and_evaluate(n_datasets=10)
