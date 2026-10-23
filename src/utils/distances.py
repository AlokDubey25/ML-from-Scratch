import numpy as np

def euclidean_distance(x1 : np.ndarray, x2 : np.ndarray) -> float:
    return np.sqrt(np.sum((x1 - x2) ** 2))      # woh direct vala distance btw two pts

def manhattan_distance(x1 : np.ndarray, x2 : np.ndarray) -> float:
    return np.sum(np.abs(x1 - x2))              # mod vala distance btw two pts

def cosine_similarity(x1 : np.ndarray, x2 : np.ndarray) -> float:
    dot = np.dot(x1, x2)
    norm = np.linalg.norm(x1) * np.linalg.norm(x2)
    return dot/norm if norm > 0 else 0.0

def cosine_distance(x1 : np.ndarray, x2 : np.ndarray) -> float:
    return 1- cosine_similarity(x1, x2)

def minkowski_distance(x1 : np.ndarray, x2 : np.ndarray, p : int) -> float:
    return np.sum(np.abs(x1-x2) ** p) ** (1/p)    