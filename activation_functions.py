import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    res = 1 / (1 + np.exp(-x))
    return res

def relu(x: np.ndarray) -> np.ndarray:
    res = np.maximum(x, 0)
    return res

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    res = np.where(x > 0, x, alpha * x)
    return res

def tanh(x: np.ndarray) -> np.ndarray:
    res = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return res

def linear(x: np.ndarray) -> np.ndarray:
    res = x
    return res
