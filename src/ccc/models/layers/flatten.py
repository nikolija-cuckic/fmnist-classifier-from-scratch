class Flatten:
    def __init__(self):
        self.in_shape = None

    def forward(self, x):
        self.in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_out):
        return grad_out.reshape(self.in_shape)

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []

