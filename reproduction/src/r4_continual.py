"""R4: 1D continual learning, 5 Gaussian peaks, 5 phases (paper Fig 3.4). KAN [1,1] G=200 vs MLP ~ equal params.
Setup (fixed a priori): Adam lr=1e-2, 1500 full-batch steps per phase, same for both; KAN grid fixed (update_grid=False)."""
import sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
C = [-0.8, -0.4, 0.0, 0.4, 0.8]; SIG = 0.07; HALF = 0.2; STEPS = 1500; LR = 1e-2
def target(x): return sum(torch.exp(-((x - c) ** 2) / (2 * SIG ** 2)) for c in C)
def region(k, n, g): return C[k] - HALF + 2 * HALF * torch.rand(n, 1, generator=g)

def build(kind, seed):
    if kind == "KAN": return KAN(width=[1, 1], grid=200, k=3, seed=seed, auto_save=False)
    return make_mlp([1, 16, 10, 1], seed)

def adam_fit(m, x, y, steps, lr):
    params = m.get_params() if hasattr(m, "get_params") else list(m.parameters())
    o = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        o.zero_grad(); l = torch.mean((m(x) - y) ** 2); l.backward(); o.step()

def run(seed):
    g = torch.Generator().manual_seed(seed)
    tr = [region(k, 200, g) for k in range(5)]; te = [region(k, 500, g) for k in range(5)]
    xs = torch.linspace(-1, 1, 400).unsqueeze(1); res = {}
    for kind in ("KAN", "MLP"):
        m = build(kind, seed); P = kan_params(m) if kind == "KAN" else mlp_params(m)
        curves, forget = [], np.zeros((5, 5))  # forget[phase, region] = test RMSE on that region after phase
        for k in range(5):
            adam_fit(m, tr[k], target(tr[k]), STEPS, LR)
            with torch.no_grad():
                curves.append(m(xs).squeeze().tolist())
                for j in range(5): forget[k, j] = rmse(m(te[j]), target(te[j]))
        # control: joint training on all five regions (is capacity the problem?)
        mj = build(kind, seed); xa = torch.cat(tr); adam_fit(mj, xa, target(xa), STEPS * 3, LR)
        with torch.no_grad(): joint = [rmse(mj(te[j]), target(te[j])) for j in range(5)]
        res[kind] = dict(params=P, curves=curves, forget=forget.tolist(), joint=joint)
    res["train_x"] = [t.squeeze().tolist() for t in tr]; res["xs"] = xs.squeeze().tolist(); return res

def analyze(allres):
    plt.rcParams.update({"font.size": 11}); r0 = allres[0]; xs = np.array(r0["xs"]); gt = target(torch.tensor(xs)).numpy()
    fig, axs = plt.subplots(3, 5, figsize=(17, 8.5), sharex=True, sharey=True)
    for k in range(5):
        a = axs[0, k]; a.plot(xs, gt, "k-", lw=1, alpha=.4)
        a.plot(r0["train_x"][k], target(torch.tensor(r0["train_x"][k])).numpy(), "k.", ms=3); a.set_title(f"phase {k+1}: data"); 
        for row, kind, col in ((1, "KAN", "C3"), (2, "MLP", "C0")):
            b = axs[row, k]; b.plot(xs, gt, "k-", lw=1, alpha=.3); b.plot(xs, r0[kind]["curves"][k], col, lw=1.5)
            mx = max(r0[kind]["curves"][k]); note = f"\n(max pred {mx:.0f}, off-scale)" if mx > 1.5 else ""
            b.set_title(f"{kind} after phase {k+1}" + note, fontsize=10)
    for r, n in enumerate(("data (seen this phase)", f"KAN [1,1] G=200 ({r0['KAN']['params']} params)", f"MLP [1,16,10,1] ({r0['MLP']['params']} params)")): axs[r, 0].set_ylabel(n, fontsize=9)
    for a in axs[2]: a.set_xlabel("x")
    axs[0, 0].set_ylim(-0.3, 1.5)  # shared y-axis clipped: MLP extrapolation can reach O(100), which would hide the peaks
    fig.suptitle("R4: continual learning, 1D, 5 Gaussian peaks presented sequentially (seed 0; grey = target)"); fig.tight_layout(); fig.savefig(f"{FIG}/r4_continual.png", dpi=150); plt.close(fig)
    import pandas as pd; rows = []
    for kind in ("KAN", "MLP"):
        F = np.array([r[kind]["forget"] for r in allres])  # seeds x phase x region
        for k in range(1, 5):
            old = F[:, k, :k].mean(1)  # mean RMSE over old regions after phase k+1
            rows.append(dict(model=kind, after_phase=k + 1, old_regions_rmse_mean=old.mean(), old_regions_rmse_std=old.std(),
                             newest_region_rmse_mean=F[:, k, k].mean()))
        rows.append(dict(model=kind, after_phase="joint-training control (all regions)", old_regions_rmse_mean=np.mean([r[kind]["joint"] for r in allres]), old_regions_rmse_std=np.std([np.mean(r[kind]["joint"]) for r in allres])))
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/r4_forgetting.csv", index=False); print(df.to_string())
    # forgetting curve of region 1
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind, col in (("KAN", "C3"), ("MLP", "C0")):
        F = np.array([r[kind]["forget"] for r in allres]); m = F[:, :, 0].mean(0); s = F[:, :, 0].std(0)
        ax.errorbar(range(1, 6), m, yerr=s, fmt="o-", color=col, capsize=3, label=kind)
    ax.set_yscale("log"); ax.set_xlabel("phase"); ax.set_ylabel("test RMSE on region 1 (peak 1)"); ax.set_title("R4: forgetting of the first region"); ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{FIG}/r4_forgetting_region1.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replot": analyze(json.load(open(f"{RES}/r4_continual.json"))); sys.exit()
    threads(2); log = logger("r4_continual"); allres = []
    for s in SEEDS: log(f"seed {s}"); allres.append(run(s))
    json.dump(allres, open(f"{RES}/r4_continual.json", "w")); analyze(allres)
