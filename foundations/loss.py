import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        #y_pred # "clip" the predicted probabilities so they aren't too large when natural-logged
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # stops ln(0) = -inf when predicted probabilities are  0 or 1
        true_class_prob = y_true * np.log(y_pred)
        wrong_class_prob = (1 - y_true) * (np.log(1 - y_pred))
        loss = -np.mean(true_class_prob + wrong_class_prob) # negate the mean log-likelihood. this turns maximum likelihood into minimizing NLL (negative log-likelihood)
        return float(np.round(loss, 4))


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        true_class_prob = y_true * np.log(y_pred)
        err_total = np.sum(true_class_prob, axis=1)
        loss = -np.mean(err_total, axis=0)
        return float(np.round(loss, 4))

