# coding:utf-8
import logging
import numpy as np
from base import BaseEstimator
from kernerls import Linear

np.random.seed(9999)

class SVM(BaseEstimator):
    def __init__(self, C=1.0, class_weight=None, kernel=None, tol=1e-3, max_iter=100):
        """SVM usando SMO simplificado, com suporte a class_weight e kernel genérico."""
        self.C = C
        self.class_weight = class_weight
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel if kernel is not None else Linear()
        self.b = 0
        self.alpha = None
        self.K = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :]).ravel()

        self.alpha = np.zeros(self.n_samples)
        self.sv_idx = np.arange(0, self.n_samples)
        support_mask = self.alpha > 1e-5
        self.support_vectors = self.X[support_mask]
        self.support_vector_labels = self.y[support_mask]
        self.support_vector_alphas = self.alpha[support_mask]

        return self._train()

    def _get_Ci(self, i):
        """Retorna o C ajustado com base na classe da amostra i."""
        if self.class_weight is not None:
            return self.C * self.class_weight.get(self.y[i], 1.0)
        else:
            return self.C

    def _find_bounds(self, i, j):
        """Encontra os limites L e H ajustados para C_i e C_j."""
        Ci = self._get_Ci(i)
        Cj = self._get_Ci(j)

        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(Cj, Ci - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - Ci)
            H = min(Ci, self.alpha[i] + self.alpha[j])

        return L, H, Ci, Cj

    def _train(self):
        iters = 0
        while iters < self.max_iter:
            iters += 1
            alpha_prev = np.copy(self.alpha)

            for j in range(self.n_samples):
                i = self.random_index(j)
                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                if eta >= 0:
                    continue

                L, H, Ci, Cj = self._find_bounds(i, j)
                e_i, e_j = self._error(i), self._error(j)
                alpha_io, alpha_jo = self.alpha[i], self.alpha[j]

                # Atualiza alpha[j]
                self.alpha[j] -= (self.y[j] * (e_i - e_j)) / eta
                self.alpha[j] = self.clip(self.alpha[j], H, L)

                # Atualiza alpha[i]
                self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_jo - self.alpha[j])

                # Atualiza o bias b
                b1 = self.b - e_i - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, i] \
                     - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j]
                b2 = self.b - e_j - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j] \
                     - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j]

                if 0 < self.alpha[i] < Ci:
                    self.b = b1
                elif 0 < self.alpha[j] < Cj:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)

            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break

        logging.info("Convergence has reached after %s." % iters)
        self.sv_idx = np.where(self.alpha > 0)[0]

    def _predict(self, X=None):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self._predict_row(X[i, :]))
        return result

    def _predict_row(self, X):
        k_v = self.kernel(self.X[self.sv_idx], X)
        return np.dot(self.alpha[self.sv_idx] * self.y[self.sv_idx], k_v.ravel()) + self.b


    def clip(self, alpha, H, L):
        return min(max(alpha, L), H)

    def _error(self, i):
        return self._predict_row(self.X[i]) - self.y[i]

    def random_index(self, z):
        i = z
        while i == z:
            i = np.random.randint(0, self.n_samples - 1)
        return i

    def decision_function(self, X):
        K = self.kernel(X, self.X)  # K.shape = (num_samples_input, num_samples_train)
        return np.dot(K, self.alpha * self.y) + self.b
