import numpy as np


# ______ Metrics for classificartions..._______

def accuracy(y_true : np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray, positive_label = 1) -> float:
    TP = np.sum((y_pred == positive_label) & (y_true == positive_label))
    FP = np.sum((y_pred == positive_label) & (y_true == positive_label))
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray, positive_label = 1) -> float:
    TP = np.sum((y_pred == positive_label) & (y_true == positive_label))
    FN = np.sum((y_pred != positive_label) & (y_true == positive_label))
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label = 1) -> float:
    P = precision(y_true, y_pred, positive_label)
    R = recall(y_true, y_pred, positive_label)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n,n), dtype = int)
    class_to_index = {c: i for i,c in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        cm[class_to_index[true], class_to_index[pred]] += 1
    return cm



# _______Metrics for regression...________

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2) 
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot> 0 else 0.0

