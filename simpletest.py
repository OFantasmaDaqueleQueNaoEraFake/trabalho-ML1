import logging
import numpy as np
from kernerls import Linear, RBF
from SVM_model import SVM
import pandas as pd



logging.basicConfig(level=logging.DEBUG)

def pca_manual(X, n_components=2):
    # Centraliza os dados
    X_meaned = X - np.mean(X, axis=0)

    # Calcula a matriz de covariância
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Calcula autovalores e autovetores
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Ordena os autovetores pelos autovalores em ordem decrescente
    sorted_idx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    eigen_values = eigen_values[sorted_idx]

    # Seleciona os n autovetores principais
    eigen_vectors_subset = eigen_vectors[:, :n_components]

    # Projeta os dados
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
        "confusion_matrix": np.array([[TN, FP], [FN, TP]])
    }

def normalize_manual(X):
    maximo = np.max(X, axis=0)
    minimo = np.minimum(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # evita divisão por zero
    return (X - mean) / std 

def classification_from_dataframe(df):
    # Define a coluna de rótulo
    y = df.iloc[:, -1].copy()       # Última coluna como y
    X = df.iloc[:, :-1].copy()      # Todas as outras colunas como X


    # Converte atributos categóricos e lida com valores ausentes
    X = pd.get_dummies(X, dummy_na=True)
    X = X.fillna(0)

    # Converte os rótulos para valores numéricos {-1, 1}
    if y.nunique() != 2:
        raise ValueError(f"Esperado problema binário. Valores encontrados em 'Class': {y.unique()}")

    y_mapped = pd.Series(np.nan, index=y.index)

    unique_classes = y.unique()
    y_mapped[y == unique_classes[0]] = -1
    y_mapped[y == unique_classes[1]] = 1

    if y_mapped.isnull().any():
        raise ValueError("Não foi possível mapear os rótulos para {-1, 1}.")

    y = y_mapped.astype(int)

    #X = normalize_manual(X.values)  # normaliza antes do PCA
    # Redução para 2D
    X_2d = pca_manual(X, n_components=2)


    # Divide treino/teste
    X = X_2d  # já está reduzido com PCA
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

    #minority_weights = [1.0, 5.0, 10.0, 20.0, 50.0]
    #kernels = [Linear(), RBF(gamma=0.001),RBF(gamma=0.01), RBF(gamma=0.1), RBF(gamma=0.5), RBF(gamma=1), RBF(gamma=10),]
    kernels = [Linear()]
    minority_weights = [1.0]
    for kernel in kernels:
        print(f"\n==> Kernel: {kernel}")
        for w in minority_weights:
            class_weight = {1: w, -1: 1.0}
            model = SVM(C=1.0, class_weight=class_weight, kernel=kernel, max_iter=500)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            metrics = compute_metrics(y_test, predictions, pos_label=1)

            print(f"\n  Class weight (1): {w}")
            print("  Accuracy :", metrics["accuracy"])
            print("  Precision:", metrics["precision"])
            print("  Recall   :", metrics["recall"])
            print("  F1-score :", metrics["f1_score"])
            print("  Confusion matrix:\n", metrics["confusion_matrix"])
            #print("Alphas da classe 1 com kernel RBF:", model.alpha[model.y == 1])
            #print("Soma total das alphas classe 1:", np.sum(model.alpha[model.y == 1]))



if __name__ == "__main__":
    for path in ["dataset_group_2_class_imbalance/dataset_38_sick.csv"]:
        df = pd.read_csv(path)
        print(df)
        classification_from_dataframe(df)