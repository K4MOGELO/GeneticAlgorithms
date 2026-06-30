"""Record an animated GIF of the foraging simulation for README embedding.

Usage:
    python scripts/record_gif.py
    python scripts/record_gif.py --config experiments/phase2/prey-predator.yml --out images/demo.gif
"""
import argparse, io, sys, tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foraging.config import resolve
from foraging.rng import make_rng
from foraging.ga_core import evolve


def render_frame(gen, tick, food, pos_alive, energy_alive, best, mean, emax,
                 nx, ny, pred_pos=None, pred_energy=None):
    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=90)
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.set_xlim(0, ny)
    ax.set_ylim(0, nx)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fr, fc = np.nonzero(food > 0)
    if fc.size:
        ax.scatter(fc, fr, s=1.5, c="white", alpha=0.55, linewidths=0, zorder=1)

    if pos_alive.size:
        e01 = np.clip(energy_alive / max(float(emax), 1e-9), 0.0, 1.0)
        cols = np.stack([1.0 - e01, e01, np.zeros_like(e01)], axis=1)
        ax.scatter(pos_alive[:, 1], pos_alive[:, 0],
                   s=10, c=cols, alpha=0.9, linewidths=0, zorder=2)

    if pred_pos is not None and pred_pos.size:
        ax.scatter(pred_pos[:, 1], pred_pos[:, 0],
                   s=18, c="dodgerblue", marker="^", alpha=0.85, linewidths=0, zorder=3)

    alive_n = int(pos_alive.shape[0]) if pos_alive.size else 0
    ax.set_title(
        f"gen {gen}  |  tick {tick}  |  alive {alive_n}  |  best {best:.1f}  mean {mean:.1f}",
        color="#cccccc", fontsize=7.5, pad=3, fontfamily="monospace",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/phase2/prey-predator.yml"))
    ap.add_argument("--out",    default=str(ROOT / "images/demo.gif"))
    ap.add_argument("--gens",       type=int, default=6,   help="generations to simulate")
    ap.add_argument("--ticks",      type=int, default=400, help="ticks per lifetime")
    ap.add_argument("--skip-ticks", type=int, default=20,  help="capture 1 frame every N ticks")
    ap.add_argument("--fps",        type=int, default=15)
    args = ap.parse_args()

    cfg = resolve("FORAGING__", args.config, [])
    cfg["ga"]["generations"]       = args.gens
    cfg["world"]["lifetime_ticks"] = args.ticks
    cfg["log"]["checkpoint_every"] = 0

    tmp = tempfile.mkdtemp(prefix="gif_run_")
    cfg["out_dir"] = tmp

    frames = []
    tick_counter = [0]
    gen_seen = set()

    def frame_cb(gen, tick, food, pos_alive, energy_alive, best, mean, emax,
                 pred_pos=None, pred_energy=None, pred_emax=None):
        tick_counter[0] += 1
        is_first  = tick == 0 and gen not in gen_seen
        is_sample = (tick_counter[0] % args.skip_ticks == 0)
        if not (is_first or is_sample):
            return
        gen_seen.add(gen)
        nx, ny = food.shape
        img = render_frame(gen, tick, food, pos_alive, energy_alive, best, mean, emax,
                           nx, ny, pred_pos=pred_pos, pred_energy=pred_energy)
        frames.append(img)
        print(f"\r  {len(frames)} frames captured  (gen {gen}  tick {tick})", end="", flush=True)

    rng = make_rng(cfg.get("seed"))
    import contextlib, os
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        evolve(cfg, rng, frame_cb=frame_cb)
    print()

    if not frames:
        print("no frames captured — check skip_ticks vs ticks")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ms_per_frame = round(1000 / args.fps)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        loop=0, duration=ms_per_frame, optimize=True,
    )
    print(f"saved {len(frames)} frames → {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
