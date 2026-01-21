import os
import time
import csv
import pickle
import numpy as np

from src.ccc.data.fmnist import get_mnist_data
from src.ccc.models.cnn_small import CNNSmall
from src.ccc.models.cnn_medium import CNNMedium
from src.ccc.models.cnn_high import CNNHigh
from src.ccc.models.cnn import CNN
from src.ccc.engine.trainer import fit
from src.ccc.engine.metrics import confusion_matrix, macro_f1
from src.ccc.utils.plot import plot_curves, plot_confusion


# Konfiguracija:
dataset_root = "data/fmnist"
model_name = "cnn"  # small, medium ili high
epochs = 3
batch_size = 64
learning_rate = 0.001
optimizer = "rmsprop"  # sgd, momentum, nag, adam
momentum = 0.9
limit_train = None   # None za ceo skup
limit_test = None    # None za ceo skup
val_split = 0.1
seed = 123
early_stop_patience = 3

# Ucitavanje podataka
(X_train, Y_train), (X_val, Y_val), (X_test, y_test) = get_mnist_data(
    root=dataset_root,
    limit_train=limit_train,
    limit_test=limit_test,
    val_split=val_split,
    seed=seed,
)

print("Dimenzije:")
print(" X_train:", X_train.shape, "Y_train:", Y_train.shape)
print(" X_val:", X_val.shape, "Y_val:", Y_val.shape)
print(" X_test:", X_test.shape, "y_test:", y_test.shape)

# Kreiranje modela
if model_name == "small":
    model = CNNSmall(num_classes=10)
elif model_name == "medium":
    model = CNNMedium(num_classes=10)
elif model_name == "high":
    model = CNNHigh(num_classes=10)
elif model_name == "cnn":
    model = CNN(num_classes=10)
else:
    print("Nepoznat model_name, koristi small")
    model = CNNSmall(num_classes=10)

# Treniranje
print("Pocetak treniranja...")
train_start = time.time()
history, best_params = fit(
    model,
    train_data=(X_train, Y_train),
    val_data=(X_val, Y_val),
    epochs=epochs,
    batch_size=batch_size,
    lr=learning_rate,
    optimizer=optimizer,
    momentum=momentum,
    early_stop_patience=early_stop_patience,
    seed=seed,
)
train_time_sec = time.time() - train_start

# Kreiranje direktorijuma za rezultate
run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(run_dir, exist_ok=True)

config_row = {
    "model_name": model_name,
    "optimizer": optimizer,
    "epochs": epochs,
    "batch_size": batch_size,
    "lr": learning_rate,
    "momentum": momentum,
    "limit_train": limit_train,
    "limit_test": limit_test,
    "val_split": val_split,
    "seed": seed,
    "early_stop_patience": early_stop_patience,
}

# Snima history u CSV (prvi red konfiguracija, poslednji red summary sa f1, acc i vremenom)
with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["config", config_row])
 
    writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "f1_macro", "train_time_sec", "test_acc"])
    for i in range(len(history.get("train_loss", []))):
        writer.writerow([
            i + 1,
            history["train_loss"][i],
            history["val_loss"][i],
            history["val_acc"][i],
            "",
            "",
            "",
        ])

plot_curves(history, run_dir)

# Ucitaj najbolje parametre u model pre testiranja
if best_params:
    params, grads = model.params_and_grads()
    for i, p in enumerate(params):
        key = f"p{i}"
        if key in best_params:
            p[...] = best_params[key]

# Test evaluacija
print("Evaluacija na test skupu...")
logits_test = []
for i in range(0, len(X_test), 256):
    xb = X_test[i:i+256]
    logits_test.append(model.forward(xb))
logits_test = np.concatenate(logits_test, axis=0)
y_pred_test = np.argmax(logits_test, axis=1)

acc_test = float(np.mean(y_pred_test == y_test))
f1_test = macro_f1(y_test, y_pred_test, num_classes=10)
cm = confusion_matrix(y_test, y_pred_test, num_classes=10)

print(f"Test accuracy: {acc_test:.4f}")
print(f"Test macro-F1: {f1_test:.4f}")

# Plot konfuziona
class_names = [str(i) for i in range(10)]
plot_confusion(cm, class_names, run_dir)

# Snima model kao pickle (struktura + tezine)
params_list, _ = model.params_and_grads()
model_artifact = {
    "model_name": model_name,
    "params": [p.copy() for p in params_list],
}

with open(os.path.join(run_dir, "best_model.pickle"), "wb") as f:
    pickle.dump(model_artifact, f)

print(f"Rezultati i model sacuvani u: {run_dir}")

# Dodaje summary red u history (f1 i acc sa testa, ukupno vreme)
with open(os.path.join(run_dir, "history.csv"), "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["summary", "", "", "", f1_test, f"{train_time_sec:.3f}", acc_test])
