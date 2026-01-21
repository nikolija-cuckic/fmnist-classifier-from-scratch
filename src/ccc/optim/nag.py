import numpy as np

def nag_step(params, grads, state: dict, lr: float, momentum: float = 0.9):
    if 'v' not in state or state['v'] is None:
        state['v'] = [np.zeros_like(p) for p in params]
    v = state['v']
    for i, (p, g) in enumerate(zip(params, grads)):
        v_prev = v[i].copy()
        v[i][...] = momentum * v[i] - lr * g
        p[...] = p - momentum * v_prev + (1 + momentum) * v[i]
