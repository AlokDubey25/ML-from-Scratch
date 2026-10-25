import numpy as np


# _____ First sigmoid activation function and its derivative _____

def sigmoid(z : np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)                   # clip to avoid overflow
    return 1.0 / (1.0 + np.exp(-z))              # sigmoid activation function

def sigmoid_derivative(z : np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


# ____ Now tanh activation function and its derivative _____

def tanh(z : np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z : np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2


# ____ Now relu activation function and its derivative _____

def relu(z : np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z : np.ndarray) -> np.ndarray:
    return np.where(z>0, 1, 0)


# ____ Now leaky relu activation function and its derivative _____

def leaky_relU(z : np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z : np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha)


#____ Now softmax activation function and its derivative _____

def softmax(z : np.ndarray) -> np.ndarray:
    if z.ndim == 1:
        z = z - np.max(z) 
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    else:
        z = z - np.max(z, axis = 1, keepdims = True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis = 1, keepdims = True)

