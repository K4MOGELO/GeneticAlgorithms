
SELECTION={}

def register(name, fn, help_text=""): 
    SELECTION[name]=fn; 
    fn._help=help_text; 
    return fn

def list_ops(): return sorted(SELECTION.keys())

def help_for(name): 
    f=SELECTION.get(name); 
    return getattr(f,"_help","") if f else ""


def tournament(parents_needed, population, fitness, rng, k=3):
    idx=[]
    n=len(population)

    for _ in range(parents_needed):
        cand=rng.integers(0,n,size=int(k))
        best=int(cand[fitness[cand].argmax()])
        idx.append(best)
    return idx

def random(parents_needed, population, fitness, rng):
    return list(rng.integers(0,len(population),size=int(parents_needed)))

def roulette(parents_needed, population, fitness, rng, eps=1e-9):
    f = fitness - fitness.min() + eps
    p = f / f.sum()
    return list(rng.choice(len(population), size=int(parents_needed), p=p))



register("tournament", tournament, "k: tournament size (int)")
register("random", random, "uniform random selection")
register("roulette", roulette, "probability depends on strength")

