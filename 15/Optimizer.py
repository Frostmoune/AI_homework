import numpy as np

# sgd动量
def sgd_momentum(w, dw, global_config, config = None):
    if config is None: 
        config = {}
    config.setdefault('v', np.zeros_like(w))

    v = config['v']

    mu = global_config['momentum']
    learning_rate = global_config['lr']
    v = mu * v - learning_rate * dw  # integrate velocity
    next_w = w + v
    config['v'] = v

    return next_w, config

# adam
def adam(x, dx, global_config, config = None):
    if config is None: 
        config = {}
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    learning_rate = global_config['lr']
    beta1 = global_config['beta1']
    beta2 = global_config['beta2']
    eps = global_config['epsilon']

    m = config['m']
    v = config['v']
    t = config['t']

    m = beta1 * m + (1 - beta1) * dx
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1 - beta2) * (dx ** 2)
    vt = v / (1 - beta2 ** t)
    next_x = x - learning_rate * mt / (np.sqrt(vt) + eps)

    config['m'] = m
    config['v'] = v
    config['t'] += 1

    return next_x, config