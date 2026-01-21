import numpy as np

def adam_step(params, grads, state: dict, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    if 'm' not in state or state.get('m') is None:
        state['m'] = [np.zeros_like(p) for p in params]
    if 'v' not in state or state.get('v') is None:
        state['v'] = [np.zeros_like(p) for p in params]
    if 't' not in state:
        state['t'] = 0
    state['t'] += 1
    t = state['t']
    m, v = state['m'], state['v']
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i][...] = beta1 * m[i] + (1 - beta1) * g
        v[i][...] = beta2 * v[i] + (1 - beta2) * (g * g)
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)
        p[...] = p - lr * m_hat / (np.sqrt(v_hat) + eps)

