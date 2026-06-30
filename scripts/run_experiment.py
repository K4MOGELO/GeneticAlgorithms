import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foraging.config import resolve
from foraging.rng import make_rng
from foraging.ga_core import evolve

cfg_path = sys.argv[1]
overrides = sys.argv[2:]
cfg = resolve("FORAGING__", cfg_path, overrides)
print("run_name:", cfg["run_name"])
print("ga:", cfg["ga"])
print("world:", {k: v for k, v in cfg["world"].items() if k != "energy"})

rng = make_rng(cfg.get("seed"))
t0 = time.time()
out_dir = evolve(cfg, rng)
print(f"done in {time.time()-t0:.1f}s -> {out_dir}")
