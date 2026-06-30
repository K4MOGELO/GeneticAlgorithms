
import time, numpy as np, matplotlib, matplotlib.pyplot as plt
from .rng import make_rng
from .ga_core import evolve

def _prefer_interactive_backend():
    if "agg" in matplotlib.get_backend().lower():
        try:
            import PyQt5; matplotlib.use("Qt5Agg", force=True)
        except Exception:
            try: matplotlib.use("TkAgg", force=True)
            except Exception: pass


def colors_from_energy(e_alive, emax):
    if e_alive.size == 0:
        return np.zeros((0,3))

    e01 = np.clip(e_alive / max(float(emax), 1e-9), 0.0, 1.0)
    return np.stack([1.0 - e01, e01, np.zeros_like(e01)], axis=1)  # red..green


def colors_from_pred_energy(e_pred, emax):
    if e_pred is None or e_pred.size == 0:
        return np.zeros((0,3))
    e01 = np.clip(e_pred / max(float(emax), 1e-9), 0.0, 1.0)
    r = np.zeros_like(e01)
    g = 0.4 * e01
    b = 0.8 + 0.2 * e01
    return np.stack([r, g, b], axis=1)

class LiveViewer:
    def __init__(self, nx, ny, fps=30):
        _prefer_interactive_backend()
        self.target_dt = 1.0 / max(1, int(fps))
        self._paused = False
        self._last = time.perf_counter()
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        self.ax.set_facecolor("black"); self.fig.patch.set_facecolor("black")
        self.ax.set_xlim(0, ny-1); self.ax.set_ylim(0, nx-1)

        pad_x = 0.5
        pad_y = 0.5
        self.ax.set_xlim(-pad_y, ny - 1 + pad_y)
        self.ax.set_ylim(-pad_x, nx - 1 + pad_x)
        
        self.ax.set_aspect("equal"); self.ax.set_xticks([]); self.ax.set_yticks([])
        self.food_scat  = self.ax.scatter([], [], s=6,  c="white", marker=".", alpha=0.9, zorder=1)
        self.agent_scat = self.ax.scatter([], [], s=18, c="cyan",  alpha=0.95, zorder=2)
        self.ttl = self.ax.set_title("", color="white")

        def on_key(e):
            if e.key == " ": self._paused = not self._paused
            elif e.key in ["+","="]: self.target_dt = max(0.005, self.target_dt * 0.8)
            elif e.key in ["-","_"]: self.target_dt = min(0.5,  self.target_dt / 0.8)
        self.fig.canvas.mpl_connect("key_press_event", on_key)

        self.pred_scat = self.ax.scatter([], [], s=22, c="blue", marker="^", alpha=0.9, zorder=3)
        plt.ion(); plt.show(block=False)



    def draw(self, gen, tick, food, pos_alive, colors_alive,
         best, mean, pred_pos=None, pred_energy=None, pred_emax=None):
        while self._paused: plt.pause(0.05)
        now = time.perf_counter(); dt = now - self._last
        if dt < self.target_dt: time.sleep(self.target_dt - dt)
        self._last = time.perf_counter()
    
        fx, fy = np.nonzero(food > 0.0)
        self.food_scat.set_offsets(np.c_[fy, fx] if fx.size else np.empty((0,2)))
        self.agent_scat.set_offsets(np.c_[pos_alive[:,1], pos_alive[:,0]] if pos_alive.size else np.empty((0,2)))
        self.agent_scat.set_color(colors_alive if colors_alive.size else np.zeros((0,3)))
    


        if pred_pos is not None and pred_energy is not None and pred_pos.size and pred_energy.size:
            alive_mask = pred_energy > 0.0
            if np.any(alive_mask):
                P = pred_pos[alive_mask]
                e = pred_energy[alive_mask]
                emax_p = float(pred_emax) if pred_emax is not None else (float(e.max()) if e.size else 1.0)
                cols_p = colors_from_pred_energy(e, emax_p)
                self.pred_scat.set_offsets(np.c_[P[:,1], P[:,0]])
                self.pred_scat.set_color(cols_p)
            else:
                self.pred_scat.set_offsets(np.empty((0,2)))
        else:
            self.pred_scat.set_offsets(np.empty((0,2)))


        self.ttl.set_text(f"gen {gen} | tick {tick} | fitness best {best:.2f} mean {mean:.2f}")
        self.fig.canvas.draw_idle(); plt.pause(0.001)
        

def run_live(cfg, fps=30):
    viewer_holder = {"viewer": None}


    def frame_cb(gen, tick, food, pos_alive, energy_alive, best, mean, emax,pred_pos=None, pred_energy=None, pred_emax=None):
        v = viewer_holder["viewer"]
        if v is None:
            nx, ny = food.shape
            v = LiveViewer(nx, ny, fps=fps)
            viewer_holder["viewer"] = v
        
        colors_foragers = colors_from_energy(energy_alive, emax)
        
        fallback_emax = float(cfg.get("world", {}).get("predators", {}).get("Emax", 10.0))
        pred_emax_eff = pred_emax if pred_emax is not None else ( float(pred_energy.max()) if (pred_energy is not None and pred_energy.size) else fallback_emax)
        
        v.draw(gen, tick, food, pos_alive, colors_foragers, best, mean,pred_pos=pred_pos, pred_energy=pred_energy, pred_emax=pred_emax_eff)




    rng = make_rng(cfg.get("seed"))
    evolve(cfg, rng, frame_cb=frame_cb)

