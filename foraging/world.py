

import numpy as np
from .world_food import FOOD
from .genome import field_names

DEFAULT_GENE_VALUES = {"speed": 0.0, "vision": 1.0, "turn": 0.0, "energy_cost": 0.0}

def gene_indices(fields):
    return {name: i for i, name in enumerate(field_names(fields))}

def wrap(v, n):
    v%=n
    if v<0: v+=n
    return v

def torus_add(xy, dxy, nx, ny):
    return np.array([wrap(xy[0]+dxy[0], nx), wrap(xy[1]+dxy[1], ny)], dtype=int)

def place_agents(n, nx, ny, rng):
    xs=rng.integers(0,nx,size=n)
    ys=rng.integers(0,ny,size=n)
    pos=np.stack([xs,ys],axis=1).astype(int)
    ang=rng.random(n)*2*np.pi
    return pos, ang

def sense_direction(food, pos, vision, nx, ny):
    fx,fy=pos
    best=None
    bestd=1e9
    r=int(max(1, vision))
    for dx in range(-r,r+1):
        x=(fx+dx)%nx
        row=food[x]
        for dy in range(-r,r+1):
            y=(fy+dy)%ny
            if row[y]>0:
                d=dx*dx+dy*dy
                if d<bestd:
                    bestd=d; best=(dx,dy)
    if best is None: return None
    return np.arctan2(best[1], best[0])

def step_agents_energy(food, pos, ang, genomes, nx, ny, rng, energy, params, gene_idx=None):
    eaten = np.zeros(len(pos), dtype=float)
    alive = energy > 0.0
    spent = np.zeros(len(pos), dtype=float)
    Emax = params.get("Emax", 10.0)
    c_b = params.get("c_b", 0.01)
    c_m = params.get("c_m", 0.001)
    c_s = params.get("c_s", 0.0)
    max_intake = params.get("max_intake", 1.0)

    gene_idx = gene_idx or {"speed": 0, "vision": 1, "turn": 2, "energy_cost": 3}
    i_speed = gene_idx.get("speed")
    i_vision = gene_idx.get("vision")
    i_turn = gene_idx.get("turn")
    i_ec = gene_idx.get("energy_cost")

    for i,g in enumerate(genomes):
        if not alive[i]:
            continue

        speed  = float(g[i_speed])  if i_speed  is not None else DEFAULT_GENE_VALUES["speed"]
        vision = float(g[i_vision]) if i_vision is not None else DEFAULT_GENE_VALUES["vision"]
        turn   = float(g[i_turn])   if i_turn   is not None else DEFAULT_GENE_VALUES["turn"]
        ec     = float(g[i_ec])     if i_ec     is not None else DEFAULT_GENE_VALUES["energy_cost"]

        dtheta = sense_direction(food, pos[i], vision, nx, ny)
        moved = False

        if dtheta is not None:
            diff=(dtheta - ang[i] + np.pi)%(2*np.pi)-np.pi
            ang[i]+= np.clip(diff, -abs(turn), abs(turn))
        else:
            ang[i]+= (rng.random()*2-1)*abs(turn)*0.5

        dx=int(round(np.cos(ang[i])*max(0.0, speed)))
        dy=int(round(np.sin(ang[i])*max(0.0, speed)))
        if dx!=0 or dy!=0: moved=True
        pos[i]=((pos[i][0]+dx)%nx, (pos[i][1]+dy)%ny)

        take = 0.0
        if food[pos[i][0], pos[i][1]]>0:
            take=min(max_intake, float(food[pos[i][0], pos[i][1]]))
            food[pos[i][0], pos[i][1]]-=take
            eaten[i]+=take

        move_cost = c_m * (speed**2 if moved else 0.0)
        sense_cost = c_s * (vision**2)
        base_cost = c_b + float(ec)
        total_cost = base_cost + move_cost + sense_cost
        dE = take - total_cost
        energy[i] = min(Emax, energy[i] + dE)
        if energy[i] <= 0.0:
            alive[i] = False
        spent[i] += total_cost

    return eaten,spent, energy, alive

def make_world(cfg, rng):
    nx=int(cfg["world"]["nx"]); ny=int(cfg["world"]["ny"])
    food_cfg=cfg["world"]["food"]; name=food_cfg.get("name","random_spots")
    kwargs={k:v for k,v in food_cfg.items() if k!="name"}
    food, reg = FOOD[name](nx,ny,rng,**kwargs)
    return {"nx":nx,"ny":ny,"food":food,"reg":reg}


def rollout(cfg, genomes, rng, gen_idx=None, frame_cb=None):
    w = make_world(cfg, rng)
    nx, ny = w["nx"], w["ny"]
    pos, ang = place_agents(len(genomes), nx, ny, rng)
    T = int(cfg["world"]["lifetime_ticks"])
    food, reg = w["food"], w["reg"]
    gene_idx = gene_indices(cfg["genome"]["fields"])

    energy_params = cfg["world"].get("energy",
        {"E0":5.0,"Emax":10.0,"c_b":0.01,"c_m":0.001,"c_s":0.0,"max_intake":1.0})
    Emax = float(energy_params.get("Emax", 10.0))
    energy = np.full(len(genomes), float(energy_params.get("E0",5.0)), dtype=float)
    total_eaten = np.zeros(len(genomes), dtype=float)
    total_spent = np.zeros(len(genomes), dtype=float)



    #preditors if they exist
    pred_cfg = cfg["world"].get("predators", {"count":0})
    n_pred = int(pred_cfg.get("count", 0))
    if n_pred > 0:
        pred_pos, pred_ang = place_agents(n_pred, nx, ny, rng)
        pred_energy = np.full(n_pred, float(pred_cfg.get("E0", 6.0)), dtype=float)
        pred_params = {
            "Emax":  float(pred_cfg.get("Emax", 10.0)),
            "c_b":   float(pred_cfg.get("c_b", 0.05)),
            "c_m":   float(pred_cfg.get("c_m", 0.02)),
            "gain":  float(pred_cfg.get("gain", 1.0)),
            "vision":int(pred_cfg.get("vision", 8)),
            "step":  int(pred_cfg.get("step", 1)),
        }
    else:
        pred_pos = pred_ang = pred_energy = pred_params = None


    for t in range(T):
        eaten, spent, energy, alive = step_agents_energy(
            food, pos, ang, genomes, nx, ny, rng, energy, energy_params, gene_idx=gene_idx)
        total_eaten += eaten
        total_spent += spent

        # predators move and eat foragers
        if n_pred > 0:
            kills, pred_energy, pred_alive, alive = step_predators(
                pred_pos, pred_ang, pred_energy, pos, alive, nx, ny, rng, pred_params
            )

        if reg: reg()

        if t == 0 and n_pred > 0:
            print(f"[gen {gen_idx}] predators: {n_pred}, first pos={pred_pos[0].tolist()}, E0={pred_energy[0]:.2f}")
        if n_pred > 0 and (t % 50 == 0):
            print(f"[gen {gen_idx} t={t}] pred_alive={int((pred_energy>0).sum())}")

        if frame_cb is not None:
            pos_alive    = pos[alive]
            energy_alive = energy[alive]
            f_now = total_eaten - total_spent
            frame_cb(gen_idx, t, food, pos_alive, energy_alive,
                     float(f_now.max()), float(f_now.mean()), Emax,
                     pred_pos=(pred_pos if n_pred>0 else None),
                     pred_energy=(pred_energy if n_pred>0 else None),
                     pred_emax=(float(pred_params["Emax"]) if n_pred>0 else None))

        if not np.any(alive):
            break

    fitness = total_eaten - total_spent
    return fitness


def step_predators(pred_pos, pred_ang, pred_energy, for_pos, for_alive,
                   nx, ny, rng, params):

    kills = np.zeros(len(pred_pos))
    alive = pred_energy > 0.0

    Emax   = float(params.get("Emax", 10.0))
    c_b    = float(params.get("c_b", 0.05))
    c_m    = float(params.get("c_m", 0.02))
    gain   = float(params.get("gain", 1.0))
    vision = int(params.get("vision", 8))
    step   = int(params.get("step", 1))

    for i in range(len(pred_pos)):
        if not alive[i]:
            continue

        # default wander
        dx = rng.integers(-1, 2)
        dy = rng.integers(-1, 2)

        # chase nearest alive forager if within vision 
        if np.any(for_alive):
            F = for_pos[for_alive]
            d = F - pred_pos[i]
            j = np.argmin(np.sum(np.abs(d), axis=1))
            if np.sum(np.abs(d[j])) <= vision:
                dx = np.sign(d[j,0]); dy = np.sign(d[j,1])

        # move on torus
        mv = abs(dx) + abs(dy)
        pred_pos[i,0] = (pred_pos[i,0] + dx * step) % nx
        pred_pos[i,1] = (pred_pos[i,1] + dy * step) % ny

        if np.any(for_alive):
            for j,(fx,fy) in enumerate(for_pos):
                if for_alive[j] and fx == pred_pos[i,0] and fy == pred_pos[i,1]:
                    for_alive[j] = False
                    kills[i] += 1
                    pred_energy[i] = min(Emax, pred_energy[i] + gain)
                    break

        pred_energy[i] -= (c_b + c_m * mv)
        if pred_energy[i] <= 0.0:
            alive[i] = False
            pred_energy[i] = 0.0
        else:
            # cap at Emax to prevent immortal snowballing
            pred_energy[i] = min(pred_energy[i], Emax)

    return kills, pred_energy, alive, for_alive

