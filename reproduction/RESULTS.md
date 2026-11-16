# KAN reproduction + extensions: results

Paper: Liu et al., *KAN: Kolmogorov-Arnold Networks* (arXiv 2404.19756). All numbers are from `results/*.csv|json`; seeds 0,1,2 unless stated. Values are mean unless stated. **Headline: the qualitative story (KAN beats MLP in few-parameter regimes, MLPs plateau, KAN ~10x slower, KAN resists forgetting better) reproduces; the paper's clean G^-3/G^-4 scaling and machine-precision symbolic recovery do not reproduce in our setup, and large-G training is unstable.**

## 1. Environment
| Item | Value |
|---|---|
| Hardware | Intel Core i7-1255U (12 logical cores), 15.7 GB RAM, no GPU. Device: `cpu` |
| OS | Windows 11 Home 10.0.26200 |
| Software | Python 3.11.0, torch 2.15.0, **pykan 0.2.8**, numpy 2.4.6, pandas 3.1.1, matplotlib 3.11.2, sympy 1.14.0 (full list: `results/pip_freeze.txt`) |
| API used | `KAN(...)`, `model.fit(dataset, opt=..., steps=..., update_grid=...)`, `model.refine(G)` (returns a new, function-preserving model), `model.prune()`, `model.auto_symbolic(lib=...)`, `model.symbolic_formula()`. The older `model.train` / `initialize_grid_from_another_model` do not exist in this version. |

## 2. Reproduction (Part A)
"Converged seeds" = seeds whose KAN [2,1,1] train RMSE at G=20 is < 1e-2 (seeds 0,1 in float32; seed 2 sits in a bad local minimum at train RMSE ≈ 0.4–0.6 for all G; this rule was applied after seeing results). "Median error" = sqrt of the median squared test error.

| Claim | Paper result | Our result | Match? | Notes |
|---|---|---|---|---|
| R1a: staircase loss curves | Visible steps at each grid extension | Clear for [2,5,1] up to G=200 (`r1_loss_curves.png`). From G=500 the train loss jumps up (≈5e-6 mean at G=200 to 1.4e-1 at G=500 and 8.7e-1 at G=1000) and test blows up. | Partial | Large-G failure in all 3 recipes: default, grid-update-first-stage-only (R1b), float64 (R1c). |
| R1b: U-shaped test loss for [2,5,1], min near G≈50 | U-shape, min ≈ G 50 | U-shape yes. Mean test RMSE: 4.0e-4 (G=10, min), 1.2e-3 (G=20), 1.7e-2 (G=50), 6.6e-2 (G=100). Per-seed argmin G = 20, 10, 10. | Partial | Minimum is at G≈10, not 50. Median error stays ≈3e-5 until G=50, so the mean-RMSE rise is driven by a few outlier test points. |
| R1c: [2,1,1] beats [2,5,1] | [2,1,1] lower | Mean RMSE, best over G: [2,1,1] 6.5e-4 (converged f32 seeds) / 5.7e-4 (f64) vs [2,5,1] 4.0e-4. Median error: [2,1,1] 9.0e-6 (f32) / ≈5e-6 (f64) vs [2,5,1] 2.7e-5. All-seed f32 [2,1,1]: 1.2e-1 (one failed seed). | Partial | No by mean RMSE; yes by median error. |
| R1d: [2,1,1] test RMSE ∝ G^-3 (mean); ~G^-4 (median) | slopes -3 and ≈-4 | Fitted slopes (G=3–50 / 3–100). **Mean RMSE:** f32 converged -0.27 / +0.51; f64 -1.5 / -0.72. **Median error:** f32 converged -2.6 / -1.9; f64 (3 seeds) -3.1 / -2.7. Slope over G=3–1000 is positive (+0.4 to +2.7). | Partial | Median error approaches G^-3 below G≈50–100 (not -4); mean RMSE does not follow G^-3. `results/r1_slopes.csv`, `r1_variants_scaling.png`. |
| R2: KAN α≈4; MLPs plateau quickly | KAN ≈ 4, MLP flat | **MLP plateau reproduced:** α = 0.15–0.5 for all depths and targets. **KAN α not reproduced:** x·y α≈3.4 but over only 3 points (72–114 params); exp: 0.43; 20D: no usable fit. KAN saturates at ~1e2–1e3 params, then degrades. Best test RMSE: x·y KAN 1.5e-4 (114 params) vs MLP 3.0e-4 (6.8e3 params); 20D KAN 3.4e-4 (294 params; median over seeds 3.3e-6 at G=20) vs MLP 9.1e-3 (8.2e3 params); exp KAN converged seeds 4.4e-4 vs MLP 1.7e-3. | Partial | KAN is better with 20–60x fewer parameters; the N^-4 slope is not demonstrated. `r2_alphas.csv`. 20D not 100D. |
| R3: recovers `1.0·e^(1.0y²+1.0·sin(3.14x))`, machine-precision loss | [2,1,1], loss ≈ machine eps | See table below. Pruned to [2,1,1] in 2/3 seeds (λ=1e-3), 0/3 (λ=1e-2). Correct formula in 3/3 (λ=1e-3) and 2/3 (λ=1e-2). Best test RMSE 1.4e-5; typical 3e-4 to 3e-3; not machine precision. | Partial | One λ=1e-3 seed and one λ=1e-2 seed hit an LBFGS NaN in the affine retrain; we fell back to the pre-retrain symbolic model (flagged). |
| R4: KAN does not forget, MLP does | KAN retains old peaks | Visually reproduced (`r4_continual.png`). Old-region RMSE after phase 5: KAN 0.24 ± 0.02 vs MLP 1.7e2 ± 1.5e2. | Partial | KAN does **not** retain perfectly: old-region RMSE is 0.19–0.24 vs 6.7e-5 for joint training; peak 1 gets distorted (overshoots to ≈2). MLP number is dominated by extrapolation blow-up (predictions up to ≈230), so it overstates forgetting. |
| R5: KAN ~10x slower than MLP at equal params | ≈10x | Per LBFGS step, KAN/MLP median 7.4x (exp), 9.4x (x·y); range 4.4x–80x. Per forward+backward pass: median 15x / 22x. Ratio grows with G: 26–45x at G=1000. | Yes | Clean serial run, 4 threads (`results/r5_timing.csv`, depth-2/3/4 MLPs matched to KAN params). |

**R3 detail** (pipeline: [2,5,1], G=5, 50 LBFGS steps with entropy reg, prune, refit, auto_symbolic with lib x, x², sin, exp, tanh, log, affine retrain). Primary config declared in advance: λ=1e-3.

| λ | seed | pruned shape | recovered formula (rounded) | test RMSE |
|---|---|---|---|---|
| 1e-3 | 0 | [2,1,1] | 0.995·exp(1.02·y² + 1.003·sin(3.142x)) + 0.005 (NaN fallback, affine not retrained) | 2.8e-3 |
| 1e-3 | 1 | [2,1,1] | 0.999·exp(1.008·y² + 1.001·sin(3.141x)) + 0.001 | 2.9e-4 |
| 1e-3 | 2 | [2,2,1] | 1.0·exp(1.001·y² + 1.0·sin(3.142x)) | 1.5e-5 |
| 1e-2 | 0 | [2,2,1] | 1.0·exp(0.978·y² + 1.0·sin(3.142x)) | 1.4e-5 |
| 1e-2 | 1 | [2,2,1] | wrong (3.07·exp(0.998·sin(3.14x) + 1.41·sin(1.4y − 1.57)) − …); NaN fallback | 1.1e-1 |
| 1e-2 | 2 | [2,2,1] | correct main term + spurious 0.013·sin(0.881x − 1.245) | 1.1e-3 |

## 3. Hypotheses (Part B): planned, not yet run
The three extension hypotheses have **not been run yet**; no results are reported here. They are planned for the final report on branches `h1-label-noise`, `h2-equal-compute` and `h3-2d-forgetting`.

- **H1, label noise:** the KAN advantage over an MLP shrinks as label noise grows, and the optimal grid size shifts lower.
- **H2, equal training time:** with Adam or an equal wall-clock budget, MLPs close a meaningful part of the gap to KANs.
- **H3, 2D forgetting:** KANs show measurable catastrophic forgetting in 2D continual learning.

## 4. Deviations from the spec / paper
- **Grid updates:** all stages use pykan's default `update_grid=True` (the pykan grid-extension tutorial default). My first attempt with `update_grid=False` broke [2,1,1] (hidden-node range [-1,2] outside the fixed grid) and was discarded. Variant R1b (grid updated only in the first stage) behaves the same at large G.
- **Precision:** everything runs in float32 (torch default). pykan's tutorials use float64; R1c repeats [2,1,1] in float64 (3 seeds). It converged on all 3 seeds, but note float64 also changes the random init, so seeds are not comparable across precisions. R2, R3 and R4 were **not** rerun in float64.
- **Steps:** [2,5,1] and [2,2,1] use 200 LBFGS steps for G ≤ 100 and only **50 for G ≥ 200** (CPU cost; 1350 total steps instead of 1800). [2,1,1] uses the full 200 per grid. One "step" = one `optimizer.step` (≤25 function evaluations).
- **R2:** 20-D target instead of 100-D, KAN [20,1,1] with G ≤ 100 only (200 steps each). The exp panel reuses the R1 [2,1,1] runs. MLP depth = number of hidden layers (2,3,4), widths {6,10,16,28,48,80}, kept if 80 ≤ params ≤ 1.2e4; 1800 LBFGS steps, **final-step** test RMSE reported. Fitted α is over mean test RMSE; the KAN fit stops at the argmin.
- **Seeds:** one seed sets both the dataset and the init, so seed-to-seed variance includes data variance. Seed 2 of [2,1,1] never leaves a local minimum, which dominates every mean involving that model; we report converged seeds and medians alongside and flag the post hoc rule.
- **R3:** both λ=1e-3 (primary, declared first) and 1e-2 are reported; a NaN fallback was added to the affine retrain; "correct formula" means the exp(≈y² + ≈sin(πx)) term is present with coefficients within a few % (extra small terms noted). Prune thresholds are pykan defaults.
- **R4:** the optimizer, learning rate and steps (Adam, 1e-2, 1500 per phase), widths and Gaussian parameters were fixed before running and not tuned. KAN grids are not updated during phases, and the 1D test regions are ±0.2 around each peak.
- **R5:** timings from a separate clean serial run, not the (contended) R2 runs.
- pykan's per-call checkpoint saving was patched off (`MultKAN.log_history`); it breaks parallel runs and does not affect training.

## 5. Figure index (`figures/`)
| File | Description |
|---|---|
| `r1_loss_curves.png` | Train/test RMSE per LBFGS step for KAN [2,5,1] and [2,1,1] (staircase; band = min/max over seeds) |
| `r1_scaling_vs_G.png` | Test RMSE (mean, median) and train RMSE vs G, with G^-2, G^-3, G^-4 reference lines |
| `r1_variants_scaling.png` | [2,1,1] converged seeds: default vs grid-update-first-only vs float64 |
| `r2_scaling_exp.png`, `r2_scaling_xy.png`, `r2_scaling_hd.png` | Test RMSE vs parameter count (log-log), KAN vs MLP depths 2/3/4, with N^-4 and N^-4/d lines |
| `r3_step1_trained.png`, `r3_step2_pruned.png`, `r3_step3_symbolic.png` | `model.plot()` at each stage (λ=1e-3, seed 0) |
| `r4_continual.png` | 3×5 grid (data / KAN / MLP) after each of 5 phases, seed 0; y-axis clipped to [-0.3, 1.5] |
| `r4_forgetting_region1.png` | Test RMSE on region 1 after each phase |
| `demo_terminal.png` | Terminal-style render of `logs/demo_output.txt` |

Other files: `demo.py` (~25 lines of code; KAN [2,5,1] → grid extension → prune → symbolic; output `logs/demo_output.txt`, formula `1.001·exp(1.0·y² + 1.0·sin(3.142x)) − 0.001`, test RMSE 3.6e-4), `run_all.sh` (end-to-end driver; not re-run as a single script, since the actual runs were launched in stages), per-run logs in `logs/`.
