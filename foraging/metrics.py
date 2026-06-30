import numpy as np

def diversity(pop):
    A=np.array(pop, dtype=float)
    return float(np.mean(np.std(A, axis=0)))


def summarize_fitness(fx):
    A=np.array(fx, dtype=float)
    return float(np.max(A)), float(np.mean(A))

