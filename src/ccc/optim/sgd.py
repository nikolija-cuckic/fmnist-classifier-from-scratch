import numpy as np

def sgd_step(params, grads, lr: float):
    for p, g in zip(params, grads):
        p[...] = p - lr * g

