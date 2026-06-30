import numpy as np

def field_names(fields): return [f["name"] for f in fields]

def bounds(fields):
    lo = np.array([f["min"] for f in fields], dtype=float)
    hi = np.array([f["max"] for f in fields], dtype=float)
    sig = np.array([f.get("sigma", 0.1) for f in fields], dtype=float)
    return lo, hi, sig

def init_population(pop_size, fields, rng):
    lo, hi, _ = bounds(fields)
    return [rng.uniform(lo, hi).astype(float) for _ in range(pop_size)]

def clamp(g, fields):
    lo, hi, _ = bounds(fields)
    g = np.minimum(np.maximum(g, lo), hi)
    return g

def as_dict(g, fields):
    names = field_names(fields)
    return {n: float(g[i]) for i, n in enumerate(names)}

