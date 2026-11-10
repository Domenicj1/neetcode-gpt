import numpy as np # for math with arrays
from numpy.typing import NDArray


class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sum = 0
        max_z = np.max(z)
        exp_logits = []
        softmax_z = []
        
        for logit in z:
            exp_z = np.exp(logit - max_z)
            sum += exp_z
            exp_logits.append(exp_z)
        for i in exp_logits:
            softmax_z.append(i/sum)
        
        return np.round(softmax_z, 4)
