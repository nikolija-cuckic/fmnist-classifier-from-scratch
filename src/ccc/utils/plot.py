import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(history: Dict[str, List[float]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    plt.figure(figsize=(10, 4))
    # Loss
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(epochs, history['train_loss'], label='train_loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Acc
    plt.subplot(1, 2, 2)
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_acc.png'))
    plt.close()

def plot_confusion(cm: np.ndarray, class_names: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Konfuziona matrica')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('Prava klasa')
    plt.xlabel('Predikcija')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()
