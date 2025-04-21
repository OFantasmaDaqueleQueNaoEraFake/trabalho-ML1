import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin

from metrics import accuracy
from kernerls import Linear, RBF
from SVM_model import SVM


def plot_decision_boundary(model, X, y, title, scaler=None, pca=None, show_margin=False, show_legend=True):
    h = 1.0
    max_grid_points = 1000
    x_min, x_max = np.percentile(X[:, 0], [1, 99])
    y_min, y_max = np.percentile(X[:, 1], [1, 99])

    range_x = np.linspace(x_min, x_max, int(np.sqrt(max_grid_points)))
    range_y = np.linspace(y_min, y_max, int(np.sqrt(max_grid_points)))
    xx, yy = np.meshgrid(range_x, range_y)

    grid = np.c_[xx.ravel(), yy.ravel()]

    if pca is not None:
        grid_original = pca.inverse_transform(grid)  # volta ao espaço de X_scaled
    else:
        grid_original = grid

    Z = model.predict(grid_original)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)

    max_plot = 1000
    if len(X) > max_plot:
        idx = np.random.choice(len(X), size=max_plot, replace=False)
        X_plot, y_plot = X[idx], y[idx]
    else:
        X_plot, y_plot = X, y

    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=plt.cm.coolwarm, edgecolors='k', s=40)

    sv_X = model.X[model.sv_idx]
    label = 'Support Vectors' if show_legend else '_nolegend_'
    plt.scatter(sv_X[:, 0], sv_X[:, 1], facecolors='none', edgecolors='black',
                s=100, linewidths=1.5, label=label)

    if show_margin and isinstance(model.kernel, Linear):
        w = ((model.alpha * model.y)[:, np.newaxis] * model.X).sum(axis=0)
        w_2d = w[:2]
        b = model.b

        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = -(w_2d[0] * x_vals + b) / w_2d[1]
        margin = 1 / np.linalg.norm(w_2d)
        y_margin_up = y_vals + margin
        y_margin_down = y_vals - margin

        plt.plot(x_vals, y_vals, 'k-', linewidth=1, label='Hiperplano')
        plt.plot(x_vals, y_margin_up, 'k--', linewidth=1, label='Margem')
        plt.plot(x_vals, y_margin_down, 'k--', linewidth=1)

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if show_legend:
        plt.legend()
    plt.grid(True)
    plt.show()



def classification_from_dataframe(df):
    # Separa y (última coluna) e X (demais colunas)
    y = df.iloc[:, -1].copy()
    X = df.iloc[:, :-1].copy()

    # Codifica atributos categóricos e preenche NaNs
    X = pd.get_dummies(X, dummy_na=True)
    X = X.fillna(0)

    # Converte rótulos para {-1, 1}
    if y.nunique() != 2:
        raise ValueError(f"Esperado problema binário. Valores encontrados: {y.unique()}")

    unique_classes = y.unique()
    y = y.map({unique_classes[0]: -1, unique_classes[1]: 1})

    # Normaliza os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplica PCA só para visualização
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # Divide para treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    for idx, kernel in enumerate([Linear(), RBF(gamma=0.5)]):
        model = SVM(C=1.0, kernel=kernel, max_iter=500)
        model.fit(X_train, y_train)

        # Testa e mostra
        predictions = model.predict(X_test)
        acc = accuracy(y_test, predictions)

        title = f"SVM com {kernel} (Acurácia: {acc:.2f})"
        # Visualiza com dados em 2D (após PCA)
        X_test_2d = pca.transform(X_test)
        plot_decision_boundary(
            model, X_test_2d, y_test.to_numpy(),
            title,
            pca=pca
            show_margin=isinstance(kernel, Linear),
            show_legend=(idx == 0)
        )


if __name__ == "__main__":
    # Substitua pelo caminho do seu arquivo
    df = pd.read_csv("dataset_group_2_class_imbalance/dataset_1065_kc3.csv")
    classification_from_dataframe(df)
