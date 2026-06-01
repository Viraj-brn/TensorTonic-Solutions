import numpy as np


def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    arr = np.array(x)
    res = 1/(1+np.exp(-1*arr)) 

    return res
    pass