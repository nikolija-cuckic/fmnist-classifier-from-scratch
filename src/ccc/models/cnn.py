import numpy as np
from .layers import (Conv2D, ReLU, AvgPool2D, Flatten, Dense,
    softmax_cross_entropy_with_logits,
)


class CNN:
    def __init__(self, num_classes: int = 10):
        self.conv1 = Conv2D(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = AvgPool2D(kernel_size=2, stride=2)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(6, 16, kernel_size=7, stride=2, padding=2)
        self.relu2 = ReLU()

        self.flatten = Flatten()

        self.fc1 = Dense(16 * 5 * 5, 16)
        self.relu3 = ReLU()
        self.fc2 = Dense(16, 32)
        self.relu4 = ReLU()
        self.fc3 = Dense(32, num_classes)

        self._last_logits = None

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        x = self.relu4.forward(x)
        x = self.fc3.forward(x)
        self._last_logits = x
        return x

    def loss_and_grad(self, targets_onehot):
        loss, d_logits = softmax_cross_entropy_with_logits(self._last_logits, targets_onehot)
        return loss, d_logits

    def backward(self, d_logits):
        dx = self.fc3.backward(d_logits)
        dx = self.relu4.backward(dx)
        dx = self.fc2.backward(dx)
        dx = self.relu3.backward(dx)
        dx = self.fc1.backward(dx)
        dx = self.flatten.backward(dx)
        dx = self.relu2.backward(dx)
        dx = self.conv2.backward(dx)
        dx = self.relu1.backward(dx)
        dx = self.pool1.backward(dx)
        dx = self.conv1.backward(dx)
        return dx

    def params_and_grads(self):
        params = []
        grads = []
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            params.extend(layer.params)
            grads.extend(layer.grads)
        return params, grads

    def zero_grads(self):
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            for g in layer.grads:
                g[...] = 0.0

