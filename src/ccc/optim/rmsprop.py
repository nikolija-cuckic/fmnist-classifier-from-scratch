import numpy as np


def rmsprop_step(params, grads, state: dict, lr: float = 1e-3, decay: float = 0.9, eps: float = 1e-8):
    if 'cache' not in state or state.get('cache') is None:
        state['cache'] = [np.zeros_like(p) for p in params]
    cache = state['cache']
    for i, (p, g) in enumerate(zip(params, grads)):
        cache[i][...] = decay * cache[i] + (1.0 - decay) * (g * g)
        p[...] = p - lr * g / (np.sqrt(cache[i]) + eps)

