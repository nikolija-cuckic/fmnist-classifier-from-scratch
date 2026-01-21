import numpy as np

def softmax_cross_entropy_with_logits(logits: np.ndarray, targets_onehot: np.ndarray):
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    eps = 1e-12
    loss = -np.sum(targets_onehot * np.log(probs + eps)) / N
    grad = (probs - targets_onehot) / N
    return loss, grad

