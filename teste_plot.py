import logging
import numpy as np
from kernerls import Linear, RBF
from SVM_model import SVM
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

def pca_manual(X, n_components=2):
    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_idx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    eigen_vectors_subset = eigen_vectors[:, :n_components]
    return np.dot(X_meaned, eigen_vectors_subset)

def normalize_manual(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def compute_metrics(y_true, y_pred, pos_label=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    TP = np.sum((y_pred == pos_label) & (y_true == pos_label))
    TN = np.sum((y_pred != pos_label) & (y_true != pos_label))
    FP = np.sum((y_pred == pos_label) & (y_true != pos_label))
    FN = np.sum((y_pred != pos_label) & (y_true == pos_label))
    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": np.array([[TN, FP], [FN, TP]])}

def plot_svm_decision(model, X, y, kernel_name, weight):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2, levels=30)
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.5)

    for label, color, name in [(-1, 'red', 'Classe -1'), (1, 'blue', 'Classe +1')]:
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], c=color, label=name, alpha=0.6)

    for i, sv in enumerate(model.support_vectors):
        edge_color = 'black' if model.y[model.alpha > 1e-5][i] == 1 else 'gray'
        plt.scatter(sv[0], sv[1], s=100, facecolors='none', edgecolors=edge_color, linewidths=2)

    plt.legend()
    plt.title(f"Kernel: {kernel_name} | Peso da Classe 1: {weight}")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def classification_from_dataframe(df):
    y = df.iloc[:, -1].copy()
    X = df.iloc[:, :-1].copy()
    X = pd.get_dummies(X, dummy_na=True).fillna(0)

    if y.nunique() != 2:
        raise ValueError(f"Esperado problema binário. Valores encontrados: {y.unique()}")
    unique_classes = y.unique()
    y_mapped = y.replace({unique_classes[0]: -1, unique_classes[1]: 1})
    y = y_mapped.astype(int)

    X = normalize_manual(X.values)
    X_2d = pca_manual(X, n_components=2)
    X_train, X_test, y_train, y_test = train_test_split_manual(X_2d, y.to_numpy(), test_size=0.2, random_state=42)

    weights = [1.0, 5.0, 10.0, 20.0, 50.0]
    gammas = [0.001, 0.01, 0.1, 0.5, 1, 10]
    kernels = [Linear()] + [RBF(gamma=g) for g in gammas]


    for kernel in kernels:
        print(f"\n==> Kernel: {kernel}")
        for w in weights:
            class_weight = {1: w, -1: 1.0}
            model = SVM(C=1.0, class_weight=class_weight, kernel=kernel, max_iter=500)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            print(f"\n  Class weight (1): {w}")
            print("  Accuracy :", metrics["accuracy"])
            print("  Precision:", metrics["precision"])
            print("  Recall   :", metrics["recall"])
            print("  F1-score :", metrics["f1_score"])
            print("  Confusion matrix:\n", metrics["confusion_matrix"])
            print("Alphas da classe 1:", model.alpha[model.y == 1])
            print("Soma total das alphas classe 1:", np.sum(model.alpha[model.y == 1]))

            plot_svm_decision(model, X_train, y_train, kernel_name=str(kernel), weight=w)

if __name__ == "__main__":
    for path in ["dataset_group_2_class_imbalance/dataset_1061_ar4.csv",
                 "dataset_group_2_class_imbalance/dataset_1064_ar6.csv",
                 "dataset_group_2_class_imbalance/dataset_1065_kc3.csv"]:
        df = pd.read_csv(path)
        print(df)
        classification_from_dataframe(df)
