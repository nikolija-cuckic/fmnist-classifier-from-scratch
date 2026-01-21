import numpy as np

def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.argmax(logits, axis=1)
    return float(np.mean(y_pred == y_true))

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom_p = tp + fp
        denom_r = tp + fn
        p = tp / denom_p if denom_p > 0 else 0.0
        r = tp / denom_r if denom_r > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

