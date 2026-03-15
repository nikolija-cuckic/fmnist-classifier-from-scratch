# CNN Clothing Classifier (Fashion-MNIST)

This project implements a convolutional neural network for clothing image classification on the Fashion-MNIST dataset, built entirely from scratch in NumPy (no PyTorch or TensorFlow). The focus is on manual implementation of layers, backpropagation, and the training loop.

## Features

- **CNN architectures:** CNNSmall, CNNMedium, CNNHigh, plus a generic CNN model.
- **Layers:** Conv2D, ReLU, MaxPool2D, AvgPool2D, GlobalAvgPool2D, BatchNorm2D, LayerNorm1D, Dense, Flatten.
- **Loss:** Softmax + cross-entropy with numerically stable implementation.
- **Optimizers:** SGD, Momentum, NAG, RMSProp, Adam.
- **Training:** Mini-batch training, validation set, metric tracking, early stopping.

## Project Structure

```
src/ccc/models/layers/   – implementation of all layers and helper functions
                           (conv2d.py, relu.py, maxpool2d.py, avg_pool2d.py,
                           global_avg_pool2d.py, batch_norm.py, layer_norm.py,
                           dense.py, loss.py, utils.py)
src/ccc/models/          – CNN architectures (cnn_small.py, cnn_medium.py,
                           cnn_high.py, cnn.py)
src/ccc/optim/           – optimizers (sgd.py, momentum.py, nag.py,
                           rmsprop.py, adam.py)
src/ccc/engine/          – training loop and metrics (trainer.py, metrics.py)
src/ccc/data/fmnist.py   – Fashion-MNIST loading, normalization, and
                           train/val/test split
```

## Dataset: Fashion-MNIST

- **Input:** 28×28×1 grayscale images, normalized to [0, 1].
- **Targets:** 10 clothing classes, encoded as one-hot vectors of length 10.
- Train/validation/test split is handled in `fmnist.py`.

## Training and Hyperparameters

Training is typically launched through functions in `trainer.py`, with configurable choices of:
- model (CNNSmall, CNNMedium, CNNHigh, CNN),
- optimizer and learning rate,
- batch size, number of epochs, and early stopping parameters.

For systematic comparison of combinations, use `sweep.py` (see below).

## sweep.py

Runs a grid search over combinations of:
- models (`MODEL_GRID`, e.g. `cnn`, `small`, `medium`, `high`),
- optimizers (`OPTIMIZER_GRID`, e.g. `sgd`, `momentum`, `nag`, `adam`, `rmsprop`),
- learning rates (`OPTIMIZER_LRS`) and batch sizes (`BATCH_GRID`).

Loads Fashion-MNIST via `get_mnist_data`, trains models using `fit(...)` from `trainer.py`, and measures:
- best validation accuracy,
- test accuracy and macro F1,
- training time.

For each run, the script:
- creates a separate folder under `runs/`,
- saves the configuration (`config.json`), training history (`history.csv`),
- best model weights (`best_model.pickle`),
- confusion matrix and plots (loss/accuracy curves).

Appends a summary of all runs to a shared `master_summary.csv` (path defined in `MASTER_CSV`).

## conclusion.ipynb

A notebook for analyzing experiment results after running `sweep.py`.

- Automatically loads all `master_summary.csv` files from the `runs/` directory and merges them into a single DataFrame.
- Computes and displays:
  - top runs by test accuracy and macro F1,
  - best combinations by model and by optimizer,
  - statistics (mean/std/max) for test accuracy grouped by optimizer and batch size.
- Generates plots:
  - accuracy vs. training time per epoch,
  - comparative bar charts for different models and optimizers.

## Why This Project Is Interesting

- The CNN is implemented in pure NumPy, including im2col convolutions and manual backpropagation through every layer.
- The project includes a complete experimental pipeline: from the training script (`sweep.py`) to the analysis notebook (`conclusion.ipynb`), demonstrating systematic comparison of architectures and optimizers on Fashion-MNIST.
