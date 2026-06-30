import numpy as np

CROSSOVER = {}

def register(name, fn, help_text=""):
    CROSSOVER[name] = fn
    fn._help = help_text
    return fn



def uniform(p1, p2, rng, rate=0.7):
    if rng.random() >= rate:
        return p1.copy() if rng.random() < 0.5 else p2.copy()

    mask = rng.integers(0, 2, size=len(p1)).astype(bool)

    c = p1.copy()
    c[mask] = p2[mask]

    return c


def one_point(p1, p2, rng, rate=0.7):
    if rng.random() >= rate:
        return p1.copy() if rng.random() < 0.5 else p2.copy()

    i = int(rng.integers(1, len(p1)))
    return np.concatenate([p1[:i], p2[i:]]).astype(float)


register("uniform", uniform, "rate: probability of recombining (vs cloning a parent)")
register("one_point", one_point, "rate: probability of recombining (vs cloning a parent)")
