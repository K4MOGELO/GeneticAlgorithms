import argparse
from .config import resolve
from .live import run_live

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = resolve("FORAGING__", args.config, args.__dict__.get("set", []))
    run_live(cfg, fps=args.fps)

if __name__ == "__main__":
    main()


