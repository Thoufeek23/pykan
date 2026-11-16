"""Shared helpers for the KAN reproduction. pykan 0.2.8 API: model.fit(...), model.refine(G)."""
import os, sys, json, time, platform
import numpy as np, torch, torch.nn as nn
import kan
from kan import KAN, create_dataset
from kan.MultKAN import MultKAN

# pykan writes a checkpoint on every fit/refine (and breaks when auto_save=False); disable for speed + parallel safety.
MultKAN.log_history = lambda self, method_name: None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG, LOG = (os.path.join(ROOT, d) for d in ("results", "figures", "logs"))
for d in (RES, FIG, LOG): os.makedirs(d, exist_ok=True)

GRIDS = [3, 5, 10, 20, 50, 100, 200, 500, 1000]
STEPS_PER_GRID = 200
SEEDS = [0, 1, 2]
DEVICE = "cpu"

def threads(n=2): torch.set_num_threads(n)

F_EXP = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
F_XY = lambda x: x[:, [0]] * x[:, [1]]
def F_HD(d):  # exp((1/100) * sum sin^2(pi x_i / 2)); the paper's 100D target (we may use d=20)
    return lambda x: torch.exp((1 / 100) * torch.sum(torch.sin(torch.pi * x / 2) ** 2, dim=1, keepdim=True))

def make_ds(f, n_var=2, seed=0, noise=0.0, ranges=(-1, 1)):
    ds = create_dataset(f, n_var=n_var, train_num=1000, test_num=1000, ranges=list(ranges), seed=seed, device=DEVICE)
    if noise > 0:
        g = torch.Generator().manual_seed(1000 + seed)
        ds["train_label"] = ds["train_label"] + noise * torch.randn(ds["train_label"].shape, generator=g)
    return ds

def kan_params(m):  # trainable parameters actually counted by the model
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

def rmse(pred, y): return float(torch.sqrt(torch.mean((pred - y) ** 2)))
def median_rmse(pred, y): return float(torch.sqrt(torch.median((pred - y) ** 2)))

def train_kan_schedule(width, ds, seed, grids=GRIDS, steps=STEPS_PER_GRID, opt="LBFGS", lr=1.0, log=None, k=3, grid_update="all"):
    """Grid-extension training. Returns per-step history + per-grid summary. Refine from previous model (no retrain)."""
    m = KAN(width=width, grid=grids[0], k=k, seed=seed, auto_save=False, device=DEVICE)
    hist = {"train": [], "test": [], "G": [], "time": []}
    summ = []
    for gi, G in enumerate(grids):
        if gi > 0: m = m.refine(G)
        t0 = time.time()
        nst = steps[G] if isinstance(steps, dict) else steps
        r = m.fit(ds, opt=opt, steps=nst, update_grid=(grid_update == "all" or (grid_update == "first" and gi == 0)), lr=lr)  # "all" = pykan default (tutorial API_7 grid ext.); adapts grid to hidden-node range
        dt = time.time() - t0
        hist["train"] += [float(v) for v in r["train_loss"]]; hist["test"] += [float(v) for v in r["test_loss"]]
        hist["G"] += [G] * nst
        with torch.no_grad():
            err2 = (m(ds["test_input"]) - ds["test_label"]) ** 2
            mean_rmse = float(torch.sqrt(err2.mean())); med_rmse = float(torch.sqrt(err2.median()))
        s = dict(G=G, params=kan_params(m), train_rmse=hist["train"][-1], test_rmse=hist["test"][-1],
                 test_rmse_mean=mean_rmse, test_rmse_median=med_rmse, sec_per_step=dt / nst, seconds=dt)
        summ.append(s)
        if log: log(f"[{width}] seed={seed} G={G} params={s['params']} train={s['train_rmse']:.3e} test={s['test_rmse']:.3e} medRMSE={med_rmse:.3e} {dt:.1f}s")
    return m, hist, summ

def make_mlp(widths, seed):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2: layers.append(nn.SiLU())
    return nn.Sequential(*layers).to(DEVICE)

def mlp_params(m): return int(sum(p.numel() for p in m.parameters()))

def train_mlp(m, ds, steps=1800, opt="LBFGS", lr=None, time_budget=None, eval_every=1, log_times=False):
    """Full-batch MLP training. LBFGS mirrors pykan's settings (history 10, strong Wolfe, max_iter=20 per 'step')."""
    x, y, xt, yt = ds["train_input"], ds["train_label"], ds["test_input"], ds["test_label"]
    if opt == "LBFGS":
        o = torch.optim.LBFGS(m.parameters(), lr=1.0 if lr is None else lr, history_size=10, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-32, tolerance_change=1e-32)
    else:
        o = torch.optim.Adam(m.parameters(), lr=lr)
    h = {"train": [], "test": [], "time": []}
    t0 = time.time()
    def closure():
        o.zero_grad(); l = torch.mean((m(x) - y) ** 2); l.backward(); return l
    for s in range(steps):
        if opt == "LBFGS": o.step(closure)
        else: closure(); o.step()
        if s % eval_every == 0 or s == steps - 1:
            with torch.no_grad():
                h["train"].append(rmse(m(x), y)); h["test"].append(rmse(m(xt), yt)); h["time"].append(time.time() - t0)
        if time_budget and time.time() - t0 > time_budget: break
    h["seconds"] = time.time() - t0; h["steps"] = s + 1
    return h

def env_info():
    import pandas, matplotlib, sympy
    return dict(os=platform.platform(), python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                pandas=pandas.__version__, matplotlib=matplotlib.__version__, sympy=sympy.__version__,
                pykan=getattr(kan, "__version__", "0.2.8"), device=DEVICE, cuda=torch.cuda.is_available(),
                cpu="12th Gen Intel(R) Core(TM) i7-1255U (12 logical cores)", ram_gb=15.7)

def logger(name):
    path = os.path.join(LOG, name + ".log")
    fh = open(path, "a", buffering=1)
    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True); fh.write(line + "\n")
    return log

def fit_slope(x, y):
    """Least-squares slope of log10 y vs log10 x."""
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    return float(np.polyfit(lx, ly, 1)[0])


def yerr(mu, sd):
    """Asymmetric error bars for log axes: lower bar clipped to 90% of the mean (std > mean otherwise runs off-axis)."""
    mu, sd = np.asarray(mu, float), np.nan_to_num(np.asarray(sd, float))
    return np.vstack([np.minimum(sd, 0.9 * mu), sd])
