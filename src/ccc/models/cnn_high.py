import numpy as np
from .layers import (
    Conv2D, BatchNorm2D, ReLU, MaxPool2D,
    GlobalAvgPool2D, Dense, softmax_cross_entropy_with_logits,
)

class CNNHigh:
    """
    Dublji CNN sa 3x3 kernelima i global avg poolingom.
    """
    def __init__(self, num_classes: int = 10):
        self.conv1 = Conv2D(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2D(32)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(64)
        self.relu2 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)  # 28->14

        self.conv3 = Conv2D(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = BatchNorm2D(64)
        self.relu3 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)  # 14->7

        self.conv4 = Conv2D(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = BatchNorm2D(64)
        self.relu4 = ReLU()

        self.gap = GlobalAvgPool2D()  # (N,64,7,7)->(N,64)
        self.fc = Dense(64, num_classes)

        self._last_logits = None

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu1.forward(x)

        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool1.forward(x)

        x = self.conv3.forward(x)
        x = self.bn3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool2.forward(x)

        x = self.conv4.forward(x)
        x = self.bn4.forward(x)
        x = self.relu4.forward(x)

        x = self.gap.forward(x)
        x = self.fc.forward(x)
        self._last_logits = x
        return x

    def loss_and_grad(self, targets_onehot):
        loss, d_logits = softmax_cross_entropy_with_logits(self._last_logits, targets_onehot)
        return loss, d_logits

    def backward(self, d_logits):
        dx = self.fc.backward(d_logits)        
        dx = self.gap.backward(dx)             

        dx = self.relu4.backward(dx)
        dx = self.bn4.backward(dx)
        dx = self.conv4.backward(dx)

        dx = self.pool2.backward(dx)
        dx = self.relu3.backward(dx)
        dx = self.bn3.backward(dx)
        dx = self.conv3.backward(dx)

        dx = self.pool1.backward(dx)
        dx = self.relu2.backward(dx)
        dx = self.bn2.backward(dx)
        dx = self.conv2.backward(dx)

        dx = self.relu1.backward(dx)
        dx = self.bn1.backward(dx)
        dx = self.conv1.backward(dx)
        return dx

    def params_and_grads(self):
        params = []
        grads = []
        for layer in [
            self.conv1, self.bn1,
            self.conv2, self.bn2,
            self.conv3, self.bn3,
            self.conv4, self.bn4,
            self.fc,
        ]:
            params.extend(layer.params)
            grads.extend(layer.grads)
        return params, grads

    def zero_grads(self):
        for layer in [
            self.conv1, self.bn1,
            self.conv2, self.bn2,
            self.conv3, self.bn3,
            self.conv4, self.bn4,
            self.fc,
        ]:
            for g in layer.grads:
                g[...] = 0.0
