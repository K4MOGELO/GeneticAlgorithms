import numpy as np
def make_rng(seed):
    return np.random.default_rng(None if seed is None else int(seed))

