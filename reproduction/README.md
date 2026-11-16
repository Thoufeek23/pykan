# KAN reproduction

Reproduction of the main claims of *KAN: Kolmogorov–Arnold Networks* (Liu et al., ICLR 2025, arXiv 2404.19756) on top of
**pykan 0.2.8**, CPU only. It covers five reproduction targets (R1-R5). Three extension hypotheses (H1-H3) are planned but **not run yet**:

| ID | What | Script |
|---|---|---|
| R1 | Grid extension: staircase loss, U-shaped test loss, [2,1,1] vs [2,5,1], G^-3 scaling | `src/r1_grid_extension.py`, `src/r1_variants.py` |
| R2 | KAN vs MLP scaling with parameter count (x·y, exp, 20-D target) | `src/r2_scaling.py` |
| R3 | Symbolic recovery of `exp(sin(πx) + y²)` | `src/r3_symbolic.py` |
| R4 | Continual learning in 1D (KAN vs MLP) | `src/r4_continual.py` |
| R5 | Training-time ratio KAN/MLP | `src/r5_timing.py` |

Planned hypotheses (to be run for the final report, each on its own branch):

| ID | Hypothesis | Branch |
|---|---|---|
| H1 | Label noise: the KAN advantage shrinks with noise, optimal G shifts lower | `h1-label-noise` |
| H2 | Equal training time: with Adam or equal wall-clock, MLPs close part of the gap | `h2-equal-compute` |
| H3 | Continual learning in 2D: KANs show measurable forgetting | `h3-2d-forgetting` |

## Setup

Python 3.11 is required. From this folder (`reproduction/`):

```bash
# Git Bash / Linux / macOS
python3.11 -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

**Quick demo (about a minute on CPU):**

```bash
python demo.py
```

It trains KAN [2,5,1] on `f(x,y) = exp(sin(πx) + y²)` with grid extension G = 3 → 10 → 20, prunes to [2,1,1], runs `auto_symbolic`,
and prints the recovered formula (about `1.001·exp(1.0·x_2² + 1.0·sin(3.142·x_1)) − 0.001`). Reference output: `logs/demo_output.txt`.

**Everything:** `run_all.sh` is a driver script that lists every job in order. It needs a bash shell (Git Bash on Windows):

```bash
bash run_all.sh        # about 3-4 h on a 12-thread CPU; set NJOBS to half your core count
```

The original runs were launched in stages not as a single `run_all.sh` invocation,
so treat the script as the documented recipe rather than a tested one-shot run. Individual scripts can also be run alone, e.g.
`python src/r3_symbolic.py 1e-3 1e-2`. Each script writes to `results/`, `figures/` and `logs/`.

## Folder layout

```
reproduction/
├── README.md           this file
├── RESULTS.md          full results tables, verdicts and deviations
├── requirements.txt    pinned dependencies (from results/pip_freeze.txt)
├── demo.py             grid extension -> prune -> symbolic formula
├── run_all.sh          driver for all experiments
├── src/                experiment scripts; common.py holds shared helpers and the log_history patch
├── results/            R1-R5 CSV/JSON outputs, summary tables, pip_freeze.txt
├── figures/            plots used in RESULTS.md
└── logs/               per-run logs and demo_output.txt
```

## Results summary

The qualitative story reproduces: KAN beats MLP in the few-parameter regime, MLPs plateau, KAN is roughly 10x slower per step,
and KAN forgets less than an MLP. The paper's clean G^-3 / G^-4 scaling and machine-precision symbolic recovery do **not**
reproduce in our setup, and training becomes unstable at large grid sizes (G ≥ 500). Details and per-claim verdicts are in [RESULTS.md](RESULTS.md).

## Changes from the paper

- **pykan 0.2.8 API.** We use `model.fit(...)` and `model.refine(G)`; the older `model.train` and `initialize_grid_from_another_model` do not exist in this version.
- **float32.** Everything runs in torch's default float32. The paper's tutorials use float64; R1 repeats [2,1,1] in float64 as a check.
- **20-D instead of 100-D** for the high-dimensional target in R2 (KAN [20,1,1], G ≤ 100), for CPU cost.
- **Fewer steps at G ≥ 200.** [2,5,1] and [2,2,1] use 50 LBFGS steps per grid at G ≥ 200 (200 for G ≤ 100).
- **`log_history` monkeypatch.** pykan writes a checkpoint on every `fit`/`refine` and breaks when `auto_save=False`, which also makes parallel runs collide.
  We disable it with `MultKAN.log_history = lambda self, name: None` in `src/common.py` (and at the top of `demo.py`). Training is unaffected; the installed package is not modified.

The full list of deviations (seeds, R3 NaN fallback, post hoc "converged seed" rule) is in section 4 of [RESULTS.md](RESULTS.md).
