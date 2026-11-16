"""Demo: KAN [2,5,1] with grid extension -> prune -> symbolic formula, on f(x,y) = exp(sin(pi x) + y^2)."""
import torch, sympy
from kan import KAN, create_dataset
from kan.MultKAN import MultKAN
MultKAN.log_history = lambda self, name: None   # skip pykan's per-step checkpoint files

torch.set_num_threads(4)
f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000, seed=0)

model = KAN(width=[2, 5, 1], grid=3, k=3, seed=0, auto_save=False)
for G in [3, 10, 20]:                                   # grid extension: refine, don't retrain
    if G > 3: model = model.refine(G)
    r = model.fit(dataset, opt="LBFGS", steps=30, lamb=1e-3, lamb_entropy=2.0)
    print(f"G={G:3d} | train RMSE {r['train_loss'][-1]:.2e} | test RMSE {r['test_loss'][-1]:.2e}")

model = model.prune()                                   # sparsify: expect [2,1,1]
print("pruned shape:", [w[0] for w in model.width])
model.fit(dataset, opt="LBFGS", steps=30)
model.auto_symbolic(lib=["x", "x^2", "sin", "exp", "tanh", "log"], verbose=1)
r = model.fit(dataset, opt="LBFGS", steps=30, update_grid=False)   # tune affine params of symbolic fns
print(f"symbolic model | test RMSE {r['test_loss'][-1]:.2e}")
expr = sympy.expand(model.symbolic_formula()[0][0], power_exp=False)
print("formula:", expr.xreplace({n: round(float(n), 3) for n in expr.atoms(sympy.Number)}))
