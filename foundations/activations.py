import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        squished = 1 / (1 + np.exp(-(z)))
        return np.round(squished, 5) 


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        relu=[]
        for component in z:
            if component <= 0:
                relu.append(0.0)
            elif component > 0:
                relu.append(np.round(float(component), 5))
        return relu
