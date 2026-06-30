import numpy as np

MUTATION={}



def register(name, fn, help_text=""): MUTATION[name]=fn; fn._help=help_text; return fn

def _gaussian(g, rng, p=0.02, sigma=0.05, per_gene_sigma=None):
    c=g.copy()
    for i in range(len(c)):

        if rng.random()<p:
            s = per_gene_sigma[i] if isinstance(per_gene_sigma,(list,tuple,np.ndarray)) else sigma
            c[i]+=rng.normal(0.0, float(s))

    return c


def _reset(g, rng, p=0.02, lo=None, hi=None):
    c=g.copy()
    for i in range(len(c)):
        if rng.random()<p:
            l = lo[i] if lo is not None else 0.0
            h = hi[i] if hi is not None else 1.0
            c[i]=rng.uniform(l,h)
    return c


def _creep(g, rng, p=0.02, step=0.02):
    c=g.copy()
    for i in range(len(c)):
        if rng.random()<p:
            c[i]+= (rng.random()*2-1)*float(step)
    return c


register("gaussian", _gaussian, "p: per-gene mutation prob; sigma/per_gene_sigma: gaussian noise scale")
register("reset", _reset, "p: per-gene prob of resampling uniformly within [lo, hi]")
register("creep", _creep, "p: per-gene prob; step: max uniform +/- nudge")
