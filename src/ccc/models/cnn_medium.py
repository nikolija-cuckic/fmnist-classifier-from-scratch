import numpy as np
from .layers import (
    Conv2D, ReLU, MaxPool2D, Flatten, Dense,
    BatchNorm2D, LayerNorm1D, softmax_cross_entropy_with_logits,
)

class CNNMedium:
    """
    Srednji CNN: tri conv bloka i dva fully-connected sloja.
    """
    def __init__(self, num_classes: int = 10):
        self.conv1 = Conv2D(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2D(16)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(32)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = Conv2D(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = BatchNorm2D(64)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2D(kernel_size=2, stride=2)

        self.flatten = Flatten()
       
        self.fc1 = Dense(64 * 3 * 3, 128)
        self.ln1 = LayerNorm1D(128)
        self.relu4 = ReLU()
        self.fc2 = Dense(128, num_classes)

        self._last_logits = None

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.conv3.forward(x)
        x = self.bn3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.ln1.forward(x)
        x = self.relu4.forward(x)
        x = self.fc2.forward(x)
        self._last_logits = x
        return x

    def loss_and_grad(self, targets_onehot):
        loss, d_logits = softmax_cross_entropy_with_logits(self._last_logits, targets_onehot)
        return loss, d_logits

    def backward(self, d_logits):
        dx = self.fc2.backward(d_logits)
        dx = self.relu4.backward(dx)
        dx = self.ln1.backward(dx)
        dx = self.fc1.backward(dx)
        dx = self.flatten.backward(dx)
        dx = self.pool3.backward(dx)
        dx = self.relu3.backward(dx)
        dx = self.bn3.backward(dx)
        dx = self.conv3.backward(dx)
        dx = self.pool2.backward(dx)
        dx = self.relu2.backward(dx)
        dx = self.bn2.backward(dx)
        dx = self.conv2.backward(dx)
        dx = self.pool1.backward(dx)
        dx = self.relu1.backward(dx)
        dx = self.bn1.backward(dx)
        dx = self.conv1.backward(dx)
        return dx

    def params_and_grads(self):
        params = []
        grads = []
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.fc1, self.ln1, self.fc2]:
            params.extend(layer.params)
            grads.extend(layer.grads)
        return params, grads

    def zero_grads(self):
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.fc1, self.ln1, self.fc2]:
            for g in layer.grads:
                g[...] = 0.0
