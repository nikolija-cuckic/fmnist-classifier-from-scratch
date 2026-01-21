import numpy as np

def momentum_step(params, grads, state: dict, lr: float, momentum: float = 0.9):
    if 'v' not in state or state['v'] is None:
        state['v'] = [np.zeros_like(p) for p in params]
    v = state['v']
    for i, (p, g) in enumerate(zip(params, grads)):
        v[i][...] = momentum * v[i] - lr * g
        p[...] = p + v[i]

