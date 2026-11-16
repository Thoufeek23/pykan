"""R2: KAN vs MLP scaling. usage: r2_scaling.py kan <fn> <seed> | mlp <fn> <seed> <depth> | analyze
fn in {exp, xy, hd}. exp's KAN [2,1,1] curve is reused from the R1 runs (same data, same seeds)."""
import sys, glob
from common import *

HD = 20  # dimension used for the high-D target (paper: 100). See RESULTS.md deviations.
FN = {"exp": dict(f=F_EXP, n=2, kan=[2, 1, 1], title="exp(sin(pi x)+y^2)", d=2),
      "xy": dict(f=F_XY, n=2, kan=[2, 2, 1], title="x*y", d=2),
      "hd": dict(f=F_HD(HD), n=HD, kan=[HD, 1, 1], title=f"exp(0.01*sum sin^2(pi x_i/2)), {HD}D", d=HD)}
KAN_GRIDS = {"xy": GRIDS, "hd": [3, 5, 10, 20, 50, 100]}
KAN_STEPS = {"xy": {G: (200 if G <= 100 else 50) for G in GRIDS}, "hd": {G: 200 for G in GRIDS}}
WIDTHS = [6, 10, 16, 28, 48, 80]; DEPTHS = [2, 3, 4]  # depth = number of hidden layers
def mlp_shape(n_in, depth, w): return [n_in] + [w] * depth + [1]

def run_kan(fn, seed):
    c = FN[fn]; log = logger(f"r2_kan_{fn}_seed{seed}"); threads(2)
    ds = make_ds(c["f"], n_var=c["n"], seed=seed)
    m, hist, summ = train_kan_schedule(c["kan"], ds, seed, grids=KAN_GRIDS[fn], steps=KAN_STEPS[fn], log=log)
    json.dump(dict(fn=fn, seed=seed, summary=summ), open(f"{RES}/r2_kan_{fn}_seed{seed}.json", "w"))

def run_mlp(fn, seed, depth):
    c = FN[fn]; log = logger(f"r2_mlp_{fn}_seed{seed}_d{depth}"); threads(2)
    ds = make_ds(c["f"], n_var=c["n"], seed=seed); out = []
    for w in WIDTHS:
        m = make_mlp(mlp_shape(c["n"], depth, w), seed); P = mlp_params(m)
        if P < 80 or P > 1.2e4: continue
        h = train_mlp(m, ds, steps=1800, opt="LBFGS")
        r = dict(depth=depth, width=w, params=P, train_rmse=h["train"][-1], test_rmse=h["test"][-1], test_rmse_min=min(h["test"]),
                 sec_per_step=h["seconds"] / h["steps"])
        out.append(r); log(f"{fn} seed={seed} d={depth} w={w} P={P} train={r['train_rmse']:.3e} test={r['test_rmse']:.3e} ({h['seconds']:.0f}s)")
    json.dump(dict(fn=fn, seed=seed, depth=depth, runs=out), open(f"{RES}/r2_mlp_{fn}_seed{seed}_d{depth}.json", "w"))

def kan_table(fn):
    import pandas as pd
    if fn == "exp": files = sorted(glob.glob(f"{RES}/r1_2-1-1_seed*.json"))
    else: files = sorted(glob.glob(f"{RES}/r2_kan_{fn}_seed*.json"))
    rows = [dict(seed=json.load(open(f))["seed"], **s) for f in files for s in json.load(open(f))["summary"]]
    return pd.DataFrame(rows)

def analyze():
    import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    alphas = []
    for fn, c in FN.items():
        kd = kan_table(fn)
        md = pd.DataFrame([dict(seed=j["seed"], **r) for f in sorted(glob.glob(f"{RES}/r2_mlp_{fn}_seed*_d*.json")) for j in [json.load(open(f))] for r in j["runs"]])
        if kd.empty: continue
        kd.to_csv(f"{RES}/r2_kan_{fn}.csv", index=False); md.to_csv(f"{RES}/r2_mlp_{fn}.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ka = kd.groupby("G").agg(P=("params", "mean"), mu=("test_rmse", "mean"), sd=("test_rmse", "std")).reset_index()
        ax.errorbar(ka.P, ka.mu, yerr=yerr(ka.mu.values, ka.sd.fillna(0).values), fmt="o-", color="C3", lw=2, capsize=3, label=f"KAN {c['kan']}, G=3..{int(ka.G.max())}")
        # KAN exponent: fit over the grid-extension regime before it saturates (params up to the argmin of mean test RMSE)
        imin = int(ka.mu.values.argmin()); sl_full = fit_slope(ka.P, ka.mu); sl_pre = fit_slope(ka.P[: imin + 1], ka.mu[: imin + 1]) if imin >= 2 else np.nan
        alphas.append(dict(fn=fn, family="KAN", alpha_to_best=-sl_pre, alpha_all=-sl_full, best_test_rmse=float(ka.mu.min())))
        for d, col in zip(DEPTHS, ("C0", "C2", "C1")):
            sub = md[md.depth == d]
            if sub.empty: continue
            a = sub.groupby("width").agg(P=("params", "mean"), mu=("test_rmse", "mean"), sd=("test_rmse", "std")).reset_index()
            ax.errorbar(a.P, a.mu, yerr=yerr(a.mu.values, a.sd.fillna(0).values), fmt="s--", color=col, capsize=3, label=f"MLP depth {d} (SiLU)")
            alphas.append(dict(fn=fn, family=f"MLP-d{d}", alpha_to_best=-fit_slope(a.P, a.mu), alpha_all=-fit_slope(a.P, a.mu), best_test_rmse=float(a.mu.min())))
        P0, y0 = ka.P.iloc[1], ka.mu.iloc[1]; xs = np.logspace(1.7, 4.3, 20)
        ax.plot(xs, y0 * (xs / P0) ** -4, "k:", label="$N^{-4}$ (KAN theory)")
        ax.plot(xs, y0 * (xs / P0) ** (-4 / c["d"]), "k-.", label=f"$N^{{-4/d}}$, d={c['d']}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("number of parameters N"); ax.set_ylabel("test RMSE")
        ax.set_title(f"R2: {c['title']}"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(f"{FIG}/r2_scaling_{fn}.png", dpi=150); plt.close(fig)
    pd.DataFrame(alphas).to_csv(f"{RES}/r2_alphas.csv", index=False); print(pd.DataFrame(alphas).to_string())

if __name__ == "__main__":
    a = sys.argv
    if a[1] == "kan": run_kan(a[2], int(a[3]))
    elif a[1] == "mlp": run_mlp(a[2], int(a[3]), int(a[4]))
    else: analyze()
