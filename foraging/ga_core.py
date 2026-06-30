import os, numpy as np
from .genome import init_population, clamp, bounds
from .metrics import diversity, summarize_fitness
from .io_utils import ensure_dir, write_csv_row, save_checkpoint, write_jsonl
from .ga_selection import SELECTION
from .ga_crossover import CROSSOVER
from .ga_mutation import MUTATION
from .world import rollout

import inspect

def evolve(cfg, rng, frame_cb=None):
    fields = cfg["genome"]["fields"]

    lo,hi,sig = bounds(fields)
    pop = init_population(cfg["ga"]["pop_size"], fields, rng)

    out_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"{cfg['run_name']}.csv")

    for gen in range(cfg["ga"]["generations"]+1):
        fit = np.array(rollout(cfg, pop, rng, gen_idx=gen, frame_cb=frame_cb), dtype=float)
        best, mean = summarize_fitness(fit)
        div = diversity(pop)

        write_csv_row(csv_path, ["gen","best","mean","diversity"], [gen,best,mean,div])
        write_jsonl(os.path.join(out_dir,"metrics.jsonl"), {"gen":gen,"best":best,"mean":mean,"diversity":div})

        ckpt_every = cfg["log"].get("checkpoint_every")
        if ckpt_every and (gen == cfg["ga"]["generations"] or gen % ckpt_every == 0):
            save_checkpoint(os.path.join(out_dir, "ckpt"), gen, pop, fit)

        if gen==cfg["ga"]["generations"]:
            break

        E=int(cfg["ga"]["elitism"])
        idx=np.argsort(-fit)

        elites=[pop[int(i)].copy() for i in idx[:E]]

        sel_cfg=selection_cfg_for_gen(cfg["ga"], gen)
        sel=SELECTION[sel_cfg["name"]]
        xo_cfg=cfg["ga"]["crossover"];
        xo=CROSSOVER[xo_cfg["name"]]
        mut_cfg=cfg["ga"]["mutation"];
        mut=MUTATION[mut_cfg["name"]]
        kids=[]

        need = cfg["ga"]["pop_size"] - E

        sign = inspect.signature(sel)
        valid = {k: v for k, v in sel_cfg.items() if k != "name" and k in sign.parameters}
        parent_idx = sel(need*2, pop, fit, rng, **valid)

        mutkw = prep_mut_kwargs(mut_cfg, lo, hi, sig)

        for i in range(0, len(parent_idx), 2):
            p1=pop[parent_idx[i]];
            p2=pop[parent_idx[i+1 if i+1<len(parent_idx) else 0]]

            c=xo(p1, p2, rng, **{k:v for k,v in xo_cfg.items() if k!="name"})
            c=mut(c, rng, **mutkw)
            c=np.minimum(np.maximum(c,lo),hi)
            kids.append(c)
            if len(elites)+len(kids)>=cfg["ga"]["pop_size"]: break
        pop = elites + kids

    return out_dir


def selection_cfg_for_gen(ga_cfg, gen):
    schedule = ga_cfg.get("selection_schedule")
    if not schedule:
        return ga_cfg["selection"]
    for stage in schedule:
        if stage.get("gen_from", 0) <= gen <= stage.get("gen_to", gen):
            return stage
    return ga_cfg["selection"]


def prep_mut_kwargs(mut_cfg, lo, hi, sig):
    name = mut_cfg.get("name", "")
    kw = {k: v for k, v in mut_cfg.items() if k not in ("name", "sigma_mode")}
    if name == "gaussian":
        if "per_gene_sigma" not in kw and mut_cfg.get("sigma_mode", "per_gene") == "per_gene":
            kw["per_gene_sigma"] = sig
        allowed = {"p", "sigma", "per_gene_sigma"}
        return {k: v for k, v in kw.items() if k in allowed}
    if name == "reset":
        kw.setdefault("lo", lo)
        kw.setdefault("hi", hi)
        allowed = {"p", "lo", "hi"}
        return {k: v for k, v in kw.items() if k in allowed}
    if name == "creep":
        allowed = {"p", "step"}
        return {k: v for k, v in kw.items() if k in allowed}
    return kw

