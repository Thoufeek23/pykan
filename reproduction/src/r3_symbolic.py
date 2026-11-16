"""R3: KAN [2,5,1] -> sparse train -> prune -> auto_symbolic -> affine retrain -> formula. usage: r3_symbolic.py [lamb ...]
Primary config (declared before running): lamb=1e-3 (pykan tutorial value), seed 0 figures. lamb=1e-2 also reported."""
import sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
LIB = ["x", "x^2", "sin", "exp", "tanh", "log"]

def one(lamb, seed, save_figs, log):
    ds = make_ds(F_EXP, seed=seed)
    m = KAN(width=[2, 5, 1], grid=5, k=3, seed=seed, auto_save=False)
    m.fit(ds, opt="LBFGS", steps=50, lamb=lamb, lamb_entropy=2.0)  # entropy reg on (default lamb_entropy=2)
    def snap(name):
        if save_figs:
            m(ds["train_input"]); m.plot(folder=f"{LOG}/_plot_tmp"); plt.savefig(f"{FIG}/{name}.png", dpi=150, bbox_inches="tight"); plt.close("all")
    snap("r3_step1_trained")
    m = m.prune(); shape = [int(w[0]) for w in m.width]
    m(ds["train_input"]); snap("r3_step2_pruned")
    # refit after pruning (pykan tutorial does a short refit), then symbolic
    m.fit(ds, opt="LBFGS", steps=50)
    m.auto_symbolic(lib=LIB, verbose=0)
    names = [[[m.symbolic_fun[l].funs_name[j][i] for i in range(m.symbolic_fun[l].in_dim)] for j in range(m.symbolic_fun[l].out_dim)] for l in range(m.depth)]
    backup = m.copy(); nan_fallback = False
    r = m.fit(ds, opt="LBFGS", steps=50, update_grid=False)  # retrain affine params of symbolic functions
    if not np.isfinite(r["train_loss"][-1]):  # LBFGS diverged -> keep the pre-retrain symbolic model, and say so
        m = backup; nan_fallback = True
    snap("r3_step3_symbolic")
    with torch.no_grad(): te = rmse(m(ds["test_input"]), ds["test_label"]); tr = rmse(m(ds["train_input"]), ds["train_label"])
    try:
        import sympy; ex = m.symbolic_formula()[0][0]
        formula = str(ex.xreplace({n: round(float(n), 3) for n in ex.atoms(sympy.Number)}))
    except Exception as e: formula = f"ERR {e}"
    flat = str(names)
    structural = (shape == [2, 1, 1]) and ("sin" in flat) and ("x^2" in flat) and ("exp" in flat)
    out = dict(lamb=lamb, seed=seed, pruned_shape=shape, formula=formula, funs=names, train_rmse=tr, test_rmse=te, structure_correct=bool(structural), affine_retrain_nan_fallback=nan_fallback)
    log(json.dumps(out)); return out

if __name__ == "__main__":
    threads(2); log = logger("r3_symbolic")
    lambs = [float(a) for a in sys.argv[1:]] or [1e-3, 1e-2]; res = []
    for lamb in lambs:
        for s in SEEDS:
            try: res.append(one(lamb, s, save_figs=(lamb == 1e-3 and s == 0), log=log))
            except Exception as e:
                import traceback; log(f"FAILED lamb={lamb} seed={s}: {traceback.format_exc()}"); res.append(dict(lamb=lamb, seed=s, error=repr(e)))
    json.dump(res, open(f"{RES}/r3_symbolic.json", "w"), indent=1)
