import os, csv, json, numpy as np


def ensure_dir(d): os.makedirs(d, exist_ok=True)


def write_csv_row(path, header, row):
    new=not os.path.exists(path)
    with open(path,"a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

def save_checkpoint(path, gen, pop, fitness):
    ensure_dir(path)
    np.savez(os.path.join(path, f"pop_G{gen:03d}.npz"), pop=np.array(pop, dtype=float), fitness=np.array(fitness, dtype=float))


def write_jsonl(path, obj):
    with open(path,"a") as f:
        f.write(json.dumps(obj)+"\n")

