import os, time, csv, json, pickle, traceback
from itertools import product
import numpy as np

from src.ccc.data.fmnist import get_mnist_data
from src.ccc.models.cnn_small import CNNSmall
from src.ccc.models.cnn_medium import CNNMedium
from src.ccc.models.cnn_high import CNNHigh
from src.ccc.models.cnn import CNN
from src.ccc.engine.trainer import fit
from src.ccc.engine.metrics import confusion_matrix, macro_f1
from src.ccc.utils.plot import plot_curves, plot_confusion


dataset_root = "data/fmnist"
val_split = 0.1
seed = 123
limit_train = None      # None za ceo dataset
limit_test  = None      # None za ceo dataset

(X_train, Y_train), (X_val, Y_val), (X_test, y_test) = get_mnist_data(
    root=dataset_root,
    limit_train=limit_train,
    limit_test=limit_test,
    val_split=val_split,
    seed=seed,
)

MODEL_GRID      = ["cnn"]      
OPTIMIZER_GRID  = ["rmsprop"]  
OPTIMIZER_LRS = {
#"sgd": [0.05],
#"momentum": [0.1, 0.01],
#"nag": [0.1, 0.01],
#"adam": [0.001],
"rmsprop": [0.001]
}
BATCH_GRID      = [32]
EPOCHS          = 10
MOMENTUM_VALUE  = 0.9
EARLY_PATIENCE  = 3

RUNS_ROOT = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(RUNS_ROOT, exist_ok=True)

MASTER_CSV = os.path.join(RUNS_ROOT, "master_summary.csv")
with open(MASTER_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_dir","model","optimizer","lr","batch","epochs",
        "val_best_acc","test_acc","test_macro_f1","train_time_sec","status","error"
    ])

def build_model(name: str, num_classes: int = 10):
    if name == "small":
        return CNNSmall(num_classes=num_classes)
    elif name == "medium":
        return CNNMedium(num_classes=num_classes)
    elif name == "high":
        return CNNHigh(num_classes=num_classes)
    elif name == "cnn":
        return CNN(num_classes=num_classes)
    else:
        raise ValueError(f"Nepoznat model: {name}")

def safe_save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

for model_name, optimizer, batch_size in product(MODEL_GRID, OPTIMIZER_GRID, BATCH_GRID):
    for lr in OPTIMIZER_LRS[optimizer]:

        stamp = time.strftime("%H%M%S")
        run_name = f"{model_name}_opt-{optimizer}_lr-{lr}_bs-{batch_size}_{stamp}"
        run_dir = os.path.join(RUNS_ROOT, run_name)
        os.makedirs(run_dir, exist_ok=True)

        cfg = {
            "model_name": model_name,
            "optimizer": optimizer,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": EPOCHS,
            "momentum": MOMENTUM_VALUE if optimizer in ("nag", "momentum") else None,
            "val_split": val_split,
            "seed": seed,
            "limit_train": limit_train,
            "limit_test": limit_test,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        status = "ok"
        err_msg = ""
        val_best_acc = None
        test_acc = None
        test_f1 = None
        train_time_sec = None

        try:
            model = build_model(model_name, num_classes=10)

            print(f"\n=== RUN: {run_name} ===")
            train_start = time.time()
            history, best_params = fit(
                model,
                train_data=(X_train, Y_train),
                val_data=(X_val, Y_val),
                epochs=EPOCHS,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                momentum=MOMENTUM_VALUE,
                early_stop_patience=EARLY_PATIENCE,
                seed=seed,
            )
            train_time_sec = time.time() - train_start

            hist_csv = os.path.join(run_dir, "history.csv")
            with open(hist_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch","train_loss","val_loss","val_acc","f1_macro","train_time_sec","test_acc"])
                for i in range(len(history.get("train_loss", []))):
                    writer.writerow([
                        i+1,
                        history["train_loss"][i],
                        history["val_loss"][i],
                        history["val_acc"][i],
                        "",
                        "",
                        "",
                    ])

            plot_curves(history, run_dir)

            if "val_acc" in history:
                val_best_acc = float(np.max(history["val_acc"]))

            if best_params:
                params, _ = model.params_and_grads()
                for i, p in enumerate(params):
                    key = f"p{i}"
                    if key in best_params:
                        p[...] = best_params[key]

            logits_test = []
            for i in range(0, len(X_test), 256):
                xb = X_test[i:i+256]
                logits_test.append(model.forward(xb))
            logits_test = np.concatenate(logits_test, axis=0)
            y_pred_test = np.argmax(logits_test, axis=1)

            test_acc = float(np.mean(y_pred_test == y_test))
            test_f1  = macro_f1(y_test, y_pred_test, num_classes=10)
            cm = confusion_matrix(y_test, y_pred_test, num_classes=10)

            plot_confusion(cm, [str(i) for i in range(10)], run_dir)

            params_list, _ = model.params_and_grads()
            artifact = {"model_name": model_name, "params": [p.copy() for p in params_list]}
            safe_save_pickle(artifact, os.path.join(run_dir, "best_model.pickle"))

            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump({
                    "val_best_acc": val_best_acc,
                    "test_acc": test_acc,
                    "test_macro_f1": test_f1,
                    "train_time_sec": train_time_sec
                }, f, indent=2)

            with open(hist_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["summary", "", "", "", test_f1, f"{train_time_sec:.3f}", test_acc])

            print(f"[OK] {run_name} | val_best_acc={val_best_acc:.4f} | "
                f"test_acc={test_acc:.4f} | f1={test_f1:.4f} | t={train_time_sec:.1f}s")

        except Exception as e:
            status = "error"
            err_msg = str(e)
            with open(os.path.join(run_dir, "error.txt"), "w") as f:
                f.write(traceback.format_exc())
            print(f"[ERR] {run_name}: {e}")

        with open(MASTER_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_dir, model_name, optimizer, lr, batch_size, EPOCHS,
                val_best_acc, test_acc, test_f1, train_time_sec, status, err_msg
            ])

        time.sleep(5)
