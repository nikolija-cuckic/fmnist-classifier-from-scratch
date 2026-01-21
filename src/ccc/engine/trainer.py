import os
import time
import pickle
from typing import Dict, Tuple, Optional
import numpy as np

from src.ccc.optim.sgd import sgd_step
from src.ccc.optim.momentum import momentum_step
from src.ccc.optim.nag import nag_step
from src.ccc.optim.adam import adam_step
from src.ccc.optim.rmsprop import rmsprop_step
from src.ccc.engine.metrics import accuracy_from_logits


def evaluate_model(model, X: np.ndarray, y_onehot_or_idx: np.ndarray, batch_size: int = 256) -> Tuple[float, float]:
    n = len(X)
    total_loss = 0.0
    total_count = 0
    correct = 0
    idx_labels = None
    is_onehot = (y_onehot_or_idx.ndim == 2)
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        yb = y_onehot_or_idx[i:i+batch_size]
        logits = model.forward(xb)
        if is_onehot:
            loss, _ = model.loss_and_grad(yb)
            total_loss += loss * len(xb)
            idx_labels = np.argmax(yb, axis=1)
        else:
            idx_labels = yb
        y_pred = np.argmax(logits, axis=1)
        correct += int(np.sum(y_pred == idx_labels))
        total_count += len(xb)
    avg_loss = (total_loss / total_count) if is_onehot else -1.0
    acc = correct / total_count if total_count > 0 else 0.0
    return float(avg_loss), float(acc)


def fit(
    model,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.01,
    optimizer: str = "sgd",
    momentum: float = 0.9,
    early_stop_patience: int = 5,
    seed: int = 123,
) -> Tuple[Dict[str, list], Dict[str, np.ndarray]]:

    rng = np.random.default_rng(seed)
    X_train, Y_train = train_data
    X_val, Y_val = val_data

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_params: Dict[str, np.ndarray] = {}
    patience = 0
    optim_state = {}

    def collect_params_and_grads():
        return model.params_and_grads()

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(X_train))
        X_train = X_train[idx]
        Y_train = Y_train[idx]

        epoch_loss = 0.0
        seen = 0
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = Y_train[i:i+batch_size]
            logits = model.forward(xb)
            loss, d_logits = model.loss_and_grad(yb)
            model.backward(d_logits)
            params, grads = collect_params_and_grads()
            if optimizer == "sgd":
                sgd_step(params, grads, lr)
            elif optimizer == "momentum":
                momentum_step(params, grads, optim_state, lr=lr, momentum=momentum)
            elif optimizer == "nag":
                nag_step(params, grads, optim_state, lr=lr, momentum=momentum)
            elif optimizer == "adam":
                adam_step(params, grads, optim_state, lr=lr)
            elif optimizer == "rmsprop":
                rmsprop_step(params, grads, optim_state, lr=lr)
            else:
                sgd_step(params, grads, lr)
            epoch_loss += loss * len(xb)
            seen += len(xb)

        avg_train_loss = epoch_loss / seen if seen > 0 else 0.0
        val_loss, val_acc = evaluate_model(model, X_val, Y_val, batch_size=batch_size)

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(f"epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            params, _ = collect_params_and_grads()
            best_params = {f"p{i}": p.copy() for i, p in enumerate(params)}
        else:
            patience += 1
            if patience >= early_stop_patience:
                print("Early stopping aktiviran zbog stagnacije val_loss")
                break

    return history, best_params
