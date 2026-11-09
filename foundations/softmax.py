import numpy as np # for math with arrays
from numpy.typing import NDArray


class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sum = 0
        max_z = np.max(z)
        exp_nums = []
        softmax_z = []
        
        for n in z:
            exp_val = np.exp(n - max_z)
            sum += exp_val
            exp_nums.append(exp_val)
        for i in exp_nums:
            softmax_z.append(i/sum)
        
        return np.round(softmax_z, 4)
