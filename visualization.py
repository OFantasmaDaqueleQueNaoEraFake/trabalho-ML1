import logging
import numpy as np
import pandas as pd
import os
from kernerls import Linear, RBF
from SVM_model import SVM
from collections import Counter
from SVM_model_modded import New_SVM
import matplotlib.pyplot as plt

#debug
def plot_decision_boundary(model, X, y, title="SVM Decision Boundary"):
    h = 0.01
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], alpha=0.2, colors=["blue", "red"])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k', s=40)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

logging.basicConfig(level=logging.DEBUG)

def compute_class_weight(y):
    counter = Counter(y)
    total = len(y)
    return {
        1: total / (2 * counter[1]),
        -1: total / (2 * counter[-1])
    }

def dynamic_C_based_on_variance(X, base_C=1.0, method='mean_var'):
    """
    Dynamically adjust C based on variance or norms of the feature matrix X.
    
    Parameters:
        X : ndarray of shape (n_samples, n_features)
        base_C : base regularization constant
        method : 'mean_var', 'max_var', or 'feature_norm'
        
    Returns:
        C : float (adjusted C)
    """
    X = np.asarray(X)
    
    if method == 'mean_var':
        var = np.var(X, axis=0)
        avg_var = np.mean(var)
        return base_C / (1.0 + avg_var)
    
    elif method == 'max_var':
        max_var = np.max(np.var(X, axis=0))
        return base_C / (1.0 + max_var)
    
    elif method == 'feature_norm':
        norms = np.linalg.norm(X, axis=1)  # L2 norm per sample
        avg_norm = np.mean(norms)
        return base_C / (1.0 + avg_norm)
    
    else:
        raise ValueError("Unknown method: choose 'mean_var', 'max_var', or 'feature_norm'")

def pca_manual(X, n_components=2):
    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_idx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    eigen_vectors_subset = eigen_vectors[:, :n_components]
    X_reduced = np.dot(X_meaned, eigen_vectors_subset)
    return X_reduced

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

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

def normalize_manual(X):
    #min_max
    maximo = np.max(X, axis=0)
    minimo = np.min(X, axis=0)
    denom = maximo - minimo
    denom[denom == 0] = 1  # avoid division by zero for constant columns
    return (X - minimo) / denom


def classification_from_dataframe(df, flag):
    y = df.iloc[:, -1].copy()
    X = df.iloc[:, :-1].copy()
    X = pd.get_dummies(X, dummy_na=True)
    X = X.fillna(X.mean())
    X = normalize_manual(X)
    
    
    if y.nunique() != 2:
        raise ValueError(f"Esperado problema binário. Valores encontrados em 'Class': {y.unique()}")

    y_mapped = pd.Series(np.nan, index=y.index)
    unique_classes = y.unique()
    y_mapped[y == unique_classes[0]] = -1
    y_mapped[y == unique_classes[1]] = 1

    if y_mapped.isnull().any():
        raise ValueError("Não foi possível mapear os rótulos para {-1, 1}.")

    y = y_mapped.astype(int)

    X_2d = pca_manual(X, n_components=2)
    X = X_2d
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

    kernel = Linear()
    class_weight = {1: 1.0, -1: 1.0}
    if flag == 0:
        model = SVM(C=1.0, class_weight=class_weight, kernel=kernel, max_iter=500)
    else: 
        class_weight = compute_class_weight(y_train)
        C = dynamic_C_based_on_variance(X_train, base_C=1.0, method='mean_var')
        model = New_SVM(C=C, class_weight=class_weight, kernel=kernel, max_iter=500)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions, pos_label=1)
    plot_decision_boundary(model, X_test, y_test, title="Decision Boundary")

    return metrics

def evaluate_folder(folder_path, flag):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            try:
                metrics = classification_from_dataframe(df, flag)
                results.append(metrics)
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Failed on {filename}: {e}")

    if not results:
        return None

    avg_metrics = {key: np.mean([res[key] for res in results]) for key in results[0]}
    return avg_metrics

if __name__ == "__main__":
    folders = ["noise_outliers"]

    for folder in folders:
        print(f"\nEvaluating folder with standard SVM: {folder}")
        avg_results = evaluate_folder(folder, 0)
        if avg_results:
            print(f"Average Metrics for {folder}:")
            for metric, value in avg_results.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"No valid datasets found in {folder}.")

        print("\n---------------------------------------") #spacing for easier compairson

        print(f"\nEvaluating folder with Modified SVM: {folder}")
        avg_results = evaluate_folder(folder, 1)
        if avg_results:
            print(f"Average Metrics for {folder}:")
            for metric, value in avg_results.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"No valid datasets found in {folder}.")

