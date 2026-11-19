import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        #prediction = np.matmul(X, weights) # could also use np.dot(X, weights) or X @ weights
        return np.round(X @ weights, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        # sq_residual=0
        # for i in range(len(ground_truth)):
        #     sq_residual += np.square(ground_truth[i] - model_prediction[i])
        
        return np.round(np.mean(np.square(ground_truth - model_prediction)), 5)
