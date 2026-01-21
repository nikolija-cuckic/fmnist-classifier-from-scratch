import numpy as np
import pandas as pd

"""
Label meaning
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
"""

def get_mnist_data(
    root: str = "data/fmnist",
    limit_train: int | None = None,
    limit_test: int | None = None,
    val_split: float = 0.1,
    seed: int = 123,
):
    train = pd.read_csv(f"{root}/fashion-mnist_train.csv",
                        nrows=limit_train if limit_train is not None else None)
    test = pd.read_csv(f"{root}/fashion-mnist_test.csv",
                       nrows=limit_test if limit_test is not None else None)

    # Train: label + 784 piksela
    y_tr = train["label"].to_numpy().astype(np.int64)
    Xtr = train.drop(columns=["label"]).to_numpy().astype(np.uint8)

    # Test: label + 784 piksela
    y_te = test["label"].to_numpy().astype(np.int64)
    Xte = test.drop(columns=["label"]).to_numpy().astype(np.uint8)

    # Reshape + normalizacija
    Xtr = (Xtr.reshape(-1, 1, 28, 28).astype(np.float32)) / 255.0
    Xte = (Xte.reshape(-1, 1, 28, 28).astype(np.float32)) / 255.0

    # One-hot za trening i validaciju (10 klasa)
    Ytr = np.zeros((y_tr.size, 10), dtype=np.float32)
    Ytr[np.arange(y_tr.size), y_tr] = 1.0

    # Shuffle pre podele
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    Xtr, Ytr = Xtr[idx], Ytr[idx]

    # Validaciona podela
    n_val = int(len(Xtr) * float(val_split))
    if n_val > 0:
        X_val, Y_val = Xtr[:n_val], Ytr[:n_val]
        X_train, Y_train = Xtr[n_val:], Ytr[n_val:]
    else:
        X_train, Y_train = Xtr, Ytr
        X_val = np.empty((0, 1, 28, 28), dtype=np.float32)
        Y_val = np.empty((0, 10), dtype=np.float32)

    return (X_train, Y_train), (X_val, Y_val), (Xte, y_te)
