import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    m, n = X.shape
    W = np.zeros(n)
    b = 0.0
    for _ in range(steps):
        z = X@W + b
        P = _sigmoid(z)
        error = P - y
        dw = (1/m)*(X.transpose()@error)
        db = (1/m)*sum(error)
        W = W - dw*lr
        b = b - db*lr
    return W,b
    # Write code here
    pass