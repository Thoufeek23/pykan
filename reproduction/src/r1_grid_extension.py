"""R1: grid extension (paper Fig 2.3). usage: r1_grid_extension.py run <w> <seed> | analyze"""
import sys, glob
from common import *

# [2,5,1]: full 200 steps for G<=100 but only 50 for G>=200 (CPU cost; see RESULTS.md deviations)
STEPS = {"2-5-1": {G: (200 if G <= 100 else 50) for G in GRIDS}, "2-1-1": {G: 200 for G in GRIDS}}

def run(wtag, seed, variant="all"):
    width = [int(v) for v in wtag.split("-")]
    if variant == "f64": torch.set_default_dtype(torch.float64)  # R1c: default recipe in float64
    pre = {"all": "r1", "first": "r1b", "f64": "r1c"}[variant]; log = logger(f"{pre}_{wtag}_seed{seed}"); threads(2)
    ds = make_ds(F_EXP, seed=seed)
    m, hist, summ = train_kan_schedule(width, ds, seed, steps=STEPS[wtag], log=log, grid_update="all" if variant == "f64" else variant)
    json.dump(dict(width=width, seed=seed, steps=STEPS[wtag], hist=hist, summary=summ),
              open(f"{RES}/{pre}_{wtag}_seed{seed}.json", "w"))

def analyze():
    import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    data = {w: [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/r1_{w}_seed*.json"))] for w in ("2-5-1", "2-1-1")}
    rows = []
    for w, runs in data.items():
        for r in runs:
            for s in r["summary"]: rows.append(dict(model=w, seed=r["seed"], **s))
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/r1_summary.csv", index=False)
    agg = df.groupby(["model", "G"])[["train_rmse", "test_rmse", "test_rmse_median"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]; agg.to_csv(f"{RES}/r1_agg.csv", index=False)
    # --- loss curves
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (w, runs) in zip(axs, data.items()):
        for key, ls, lab in (("train", "-", "train"), ("test", "--", "test")):
            L = min(len(r["hist"][key]) for r in runs); arr = np.array([r["hist"][key][:L] for r in runs])
            mu = arr.mean(0); ax.plot(mu, ls, label=f"{lab} (mean of {len(runs)} seeds)")
            ax.fill_between(range(L), arr.min(0), arr.max(0), alpha=0.2)
        ax.set_yscale("log"); ax.set_xlabel("LBFGS step"); ax.set_title(f"KAN [{w.replace('-', ',')}], k=3"); ax.legend(); ax.grid(alpha=.3)
    axs[0].set_ylabel("RMSE"); fig.suptitle("R1: grid extension G=3..1000 (staircase; band=min/max over seeds)")
    fig.tight_layout(); fig.savefig(f"{FIG}/r1_loss_curves.png"); plt.close(fig)
    # --- scaling vs G
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    out = {}
    for ax, (w, nm) in zip(axs, (("2-1-1", "[2,1,1]"), ("2-5-1", "[2,5,1]"))):
        sub = df[df.model == w]; g = sorted(sub.G.unique())
        for col, mk, lab in (("test_rmse", "o-", "test RMSE (mean sq. err.)"), ("test_rmse_median", "s--", "test sqrt(median sq. err.)"), ("train_rmse", "^:", "train RMSE")):
            a = np.array([sub[sub.G == G][col].values for G in g]); mu, sd = a.mean(1), a.std(1)
            ax.errorbar(g, mu, yerr=yerr(mu, sd), fmt=mk, capsize=3, label=lab)
        ref = df[(df.model == w) & (df.G == 10)].test_rmse.mean()
        for p, c in ((2, "gray"), (3, "k"), (4, "r")):
            ax.plot(g, ref * (np.array(g) / 10.0) ** -p, "-", color=c, alpha=.35, lw=1.2, label=f"G^-{p} ref")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("grid size G"); ax.set_ylabel("RMSE"); ax.set_title(f"KAN {nm}"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
        if w == "2-1-1":
            for col in ("test_rmse", "test_rmse_median"):
                for lo, hi in ((3, 100), (3, 1000), (10, 200)):
                    ss = sub[(sub.G >= lo) & (sub.G <= hi)].groupby("G")[col].mean()
                    out[f"{col}_slope_G{lo}-{hi}"] = fit_slope(ss.index, ss.values)
        else:
            mt = sub.groupby("G").test_rmse.mean(); out["251_argmin_G_testmean"] = int(mt.idxmin())
            out["251_per_seed_argmin_G"] = [int(sub[sub.seed == s].set_index("G").test_rmse.idxmin()) for s in sorted(sub.seed.unique())]
    fig.suptitle("R1: test RMSE vs grid size G (log-log; mean±std over 3 seeds)"); fig.tight_layout(); fig.savefig(f"{FIG}/r1_scaling_vs_G.png"); plt.close(fig)
    # claim (c)
    for G in GRIDS:
        a = df[(df.model == "2-1-1") & (df.G == G)].test_rmse.mean(); b = df[(df.model == "2-5-1") & (df.G == G)].test_rmse.mean()
        out[f"ratio_251_over_211_G{G}"] = float(b / a)
    out["best_test_211"] = float(df[df.model == "2-1-1"].groupby("G").test_rmse.mean().min())
    out["best_test_251"] = float(df[df.model == "2-5-1"].groupby("G").test_rmse.mean().min())
    json.dump(out, open(f"{RES}/r1_claims.json", "w"), indent=1); print(json.dumps(out, indent=1)); print(agg.to_string())

if __name__ == "__main__":
    if sys.argv[1] == "run": run(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "all")
    else: analyze()
