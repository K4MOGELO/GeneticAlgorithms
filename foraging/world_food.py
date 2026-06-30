# foraging/world_food.py
import numpy as np
FOOD={}

def register(name, fn, help_text=""):
    FOOD[name]=fn; 
    fn._help=help_text;
    return fn

def _uniform(nx, ny, rng, level=0.5):
    F=np.full((nx,ny), float(level), dtype=np.float32)
    return F, None

def _random_spots(nx, ny, rng, n_spots=600, per_spot=1.0, p_regrow=0.0):
    F=np.zeros((nx,ny), dtype=np.float32)
    xs=rng.integers(0,nx,size=int(n_spots))
    ys=rng.integers(0,ny,size=int(n_spots))
    for x,y in zip(xs,ys): F[x,y]+=float(per_spot)
    reg=None

    if p_regrow and p_regrow>0.0:
        def step():
            mask=rng.random((nx,ny))<float(p_regrow)
            F[mask]+=per_spot
        reg=step
    return F, reg


register("uniform", _uniform, "level: constant food per cell")
register("random_spots", _random_spots, "n_spots, per_spot, p_regrow")
