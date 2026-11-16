"""R5: training cost, KAN vs MLP at matched parameter counts. Run ALONE (no parallel jobs) for clean timings.
Measures (a) seconds per LBFGS 'step' (one optimizer.step, <=25 function evals, same unit as the R1/R2 step counts) and
(b) seconds per forward+backward pass (removes LBFGS line-search variability). Same data (1000 pts), same thread count."""
import pandas as pd
from common import *

THREADS = 4; NSTEP = 15; NFB = 30


def time_lbfgs(model, params, ds):
    x, y = ds["train_input"], ds["train_label"]
    o = torch.optim.LBFGS(params, lr=1.0, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)

    def cl():
        o.zero_grad(); l = torch.mean((model(x) - y) ** 2); l.backward(); return l
    o.step(cl)  # warm-up
    t = time.time()
    for _ in range(NSTEP): o.step(cl)
    return (time.time() - t) / NSTEP


def time_fb(model, ds):
    x, y = ds["train_input"], ds["train_label"]
    for _ in range(3): model.zero_grad(); torch.mean((model(x) - y) ** 2).backward()
    t = time.time()
    for _ in range(NFB): model.zero_grad(); torch.mean((model(x) - y) ** 2).backward()
    return (time.time() - t) / NFB


def match_mlp(n_in, P, depth=3):
    best = min(range(2, 400), key=lambda w: abs(mlp_params(make_mlp([n_in] + [w] * depth + [1], 0)) - P))
    return best


if __name__ == "__main__":
    threads(THREADS); log = logger("r5_timing"); rows = []
    for fn, f, n, kw in (("exp", F_EXP, 2, [2, 1, 1]), ("xy", F_XY, 2, [2, 2, 1])):
        ds = make_ds(f, n_var=n, seed=0)
        for G in (3, 10, 20, 50, 100, 200, 500, 1000):
            m = KAN(width=kw, grid=G, k=3, seed=0, auto_save=False); m(ds["train_input"])
            P = kan_params(m); ks = time_lbfgs(m, m.get_params(), ds); kf = time_fb(m, ds)
            for depth in (2, 3, 4):
                w = match_mlp(n, P, depth); mm = make_mlp([n] + [w] * depth + [1], 0)
                ms = time_lbfgs(mm, list(mm.parameters()), ds); mf = time_fb(mm, ds)
                rows.append(dict(fn=fn, kan_shape=str(kw), G=G, kan_params=P, mlp_depth=depth, mlp_width=w, mlp_params=mlp_params(mm),
                                 kan_sec_per_step=ks, mlp_sec_per_step=ms, step_ratio_kan_over_mlp=ks / ms,
                                 kan_sec_per_fwdbwd=kf, mlp_sec_per_fwdbwd=mf, fwdbwd_ratio_kan_over_mlp=kf / mf))
                log(json.dumps(rows[-1]))
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/r5_timing.csv", index=False)
    print(df.groupby("fn")[["step_ratio_kan_over_mlp", "fwdbwd_ratio_kan_over_mlp"]].agg(["median", "min", "max"]).to_string())
