"""R1 variants + slope fits. Variants of the [2,1,1] grid-extension run:
  default  = pykan default grid updates, float32 (r1_*)
  first    = grid adapted only in the G=3 stage, float32 (r1b_*)
  float64  = pykan default grid updates, float64 (r1c_*)
'Converged seed' (objective rule, applied post hoc): train RMSE at G=20 < 1e-2 (i.e. the net found the right structure).
Slopes: least-squares slope of log10(error) vs log10(G) over the stated G range (negative = error falls with G)."""
import glob
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

PREFIX = {"default (float32)": "r1_2-1-1", "grid-update first stage only (float32)": "r1b_2-1-1", "default recipe, float64": "r1c_2-1-1"}


def load():
    rows = []
    for name, pre in PREFIX.items():
        for f in sorted(glob.glob(f"{RES}/{pre}_seed*.json")):
            j = json.load(open(f)); rows += [dict(variant=name, seed=j["seed"], **s) for s in j["summary"]]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load(); out = []
    conv = {(v, s): bool(df[(df.variant == v) & (df.seed == s) & (df.G == 20)].train_rmse.iloc[0] < 1e-2) for v, s in df[["variant", "seed"]].drop_duplicates().itertuples(index=False)}
    df["converged"] = [conv[(v, s)] for v, s in zip(df.variant, df.seed)]
    df.to_csv(f"{RES}/r1_variants_all.csv", index=False)
    for v in PREFIX:
        for subset in ("all seeds", "converged seeds"):
            d = df[df.variant == v] if subset == "all seeds" else df[(df.variant == v) & df.converged]
            if d.empty: continue
            for col in ("test_rmse", "test_rmse_median"):
                m = d.groupby("G")[col].mean()
                for lo, hi in ((3, 50), (3, 100), (3, 1000)):
                    s = m[(m.index >= lo) & (m.index <= hi)]
                    out.append(dict(variant=v, seeds=subset, n_seeds=d.seed.nunique(), metric=col, G_range=f"{lo}-{hi}", slope=fit_slope(s.index, s.values),
                                    best_G=int(m.idxmin()), best_value=float(m.min())))
    t = pd.DataFrame(out); t.to_csv(f"{RES}/r1_slopes.csv", index=False); pd.set_option("display.width", 220); print(t.to_string())
    print({k: v for k, v in conv.items()})
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, metric, ttl in zip(axs, ("test_rmse", "test_rmse_median"), ("test RMSE (mean sq. error)", "test sqrt(median sq. error)")):
        for (v, col) in zip(PREFIX, ("C0", "C1", "C2")):
            d = df[(df.variant == v) & df.converged]
            if d.empty: continue
            a = d.groupby("G")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(a.G, a["mean"], yerr=yerr(a["mean"].values, a["std"].values), fmt="o-", color=col, capsize=3, label=f"{v} (n={d.seed.nunique()} converged seeds)")
        g = np.array(sorted(df.G.unique())); ref = df[(df.G == 10) & df.converged][metric].mean()
        for p, c in ((2, "0.7"), (3, "0.4"), (4, "r")): ax.plot(g, ref * (g / 10.0) ** -p, color=c, alpha=.45, label=f"G^-{p}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("grid size G"); ax.set_title(f"KAN [2,1,1]: {ttl}"); ax.grid(alpha=.3, which="both")
    axs[0].set_ylabel("RMSE"); axs[0].legend(fontsize=7)
    fig.suptitle("R1 variants (converged seeds only): same degradation at large G in all three recipes?"); fig.tight_layout(); fig.savefig(f"{FIG}/r1_variants_scaling.png", dpi=150)
