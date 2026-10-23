import numpy as np

# first for regression we have losses - mse, mae and gradient loss

def mse_loss(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)          # mean squared error loss 

def mae_loss(y_true :  np.ndarray, y_pred : np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))        # mean absolute error loss

def mse_gradient(y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    n = len(y_true)
    return (2/n) * (y_pred - y_true)                      # gradient of mse loss wrt y_pred

def mae_gradient(y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    n = len(y_true)
    return (1/n) * np.sign(y_pred - y_true)               # gradient of mae loss



# now for classification we have losses - binary, categorical and hinge loss

def binary_cross_entropy(y_true : np.ndarray, y_pred : np.ndarray, eps: float = 1e-15) -> float:
    y_pred = np.clip(y_pred, eps, 1-eps)    # clip predictions to avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))       # binary cross entropy loss

def binary_cross_entropy_gradient(y_true : np.ndarray, y_pred : np.ndarray, eps: float = 1e-15) -> np.ndarray:
    y_pred = np.clip(y_pred, eps, 1-eps)    # clip predictions to avoid division by zero
    n = len(y_true)
    return (1/n) * (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred))   # gradient of binary cross entropy loss

def categorical_cross_entropy(y_true: np.ndarray, y_pred : np.ndarray, eps: float = 1e-15) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))   # categorical cross entropy loss

def hinge_loss(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    return np.mean(np.maximum(0, 1 - y_true *  y_pred))  # hinge loss

