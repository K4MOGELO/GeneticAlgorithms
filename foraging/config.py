import os, yaml, json
def deep_get(d, path, default=None):
    cur=d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur: return default
        cur=cur[p]
    return cur
def deep_set(d, path, value):
    cur=d
    parts=path.split('.')
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict): cur[p]={}
        cur=cur[p]
    cur[parts[-1]]=value
def parse_scalar(s):
    if s.lower() in ['true','false']: return s.lower()=='true'
    try:
        if '.' in s: return float(s)
        return int(s)
    except: return s
def load_yaml(path):
    with open(path,'r') as f: return yaml.safe_load(f) or {}
def merge(a,b):
    for k,v in b.items():
        if isinstance(v,dict) and isinstance(a.get(k),dict): merge(a[k],v)
        else: a[k]=v
    return a
def defaults():
    return {
        "run_name":"run",
        "seed":7,
        "out_dir":"./runs",
        "world":{
            "nx":120,"ny":150,"torus":True,"lifetime_ticks":1000,
            "food":{"name":"random_spots","n_spots":600,"per_spot":1.0,"p_regrow":0.0},
            "energy":{"E0":5.0,"Emax":10.0,"c_b":0.01,"c_m":0.001,"c_s":0.0,"max_intake":1.0}
        },



        "genome":{"fields":[
            {"name":"speed","min":0.0,"max":4.0,"sigma":0.2},
            {"name":"vision","min":1.0,"max":10.0,"sigma":0.4},
            {"name":"turn","min":-1.0,"max":1.0,"sigma":0.1},
            {"name":"energy_cost","min":0.0,"max":1.0,"sigma":0.05},
        ]},
        "ga":{"pop_size":100,"generations":50,"elitism":2,
              "selection":{"name":"tournament","k":3},
              "crossover":{"name":"uniform","rate":0.7},
              "mutation":{"name":"gaussian","p":0.02,"sigma_mode":"per_gene"}},
        "log":{"csv_every":1,"checkpoint_every":10}
    }
def resolve(env_prefix, cfg_path, overrides):
    cfg=defaults()
    if cfg_path: merge(cfg, load_yaml(cfg_path))
    for k,v in os.environ.items():
        if not k.startswith(env_prefix): continue
        key=k[len(env_prefix):].lower().replace('__','.')
        deep_set(cfg, key, parse_scalar(v))
    for ov in overrides:
        if '=' not in ov: continue
        k,v=ov.split('=',1)
        deep_set(cfg, k, parse_scalar(v))
    return cfg
def save_manifest(path, cfg, extra=None):
    os.makedirs(path, exist_ok=True)
    m={"config":cfg}
    if extra: m.update(extra)
    with open(os.path.join(path,"manifest.json"),"w") as f: json.dump(m,f,indent=2)

