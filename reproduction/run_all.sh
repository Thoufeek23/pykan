#!/usr/bin/env bash
# Reproduces everything end to end. Tested on Windows 11 / Git Bash, CPU only (12 logical cores, 15.7 GB RAM).
# Expect roughly 3-4 hours wall-clock on that machine. Jobs run in parallel (set NJOBS to your core count / 2).
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/Scripts/python.exe}; [ -x "$PY" ] || PY=.venv/bin/python
NJOBS=${NJOBS:-6}
run_par() { xargs -P "$NJOBS" -I{} sh -c "$PY src/{} > /dev/null 2>&1"; }   # logs go to logs/*.log via the scripts

# 0. environment
# python -m venv .venv && $PY -m pip install pykan torch numpy matplotlib sympy pandas scikit-learn pyyaml
$PY -m pip freeze > results/pip_freeze.txt

# Stage 1: R1 (pykan default grid updates), R1b (grid adapted in first stage only), R2 KAN + MLP sweeps, R3, R4
{
  for w in 2-1-1 2-5-1; do for s in 0 1 2; do echo "r1_grid_extension.py run $w $s"; done; done
  for s in 0 1 2; do echo "r1_grid_extension.py run 2-1-1 $s first"; echo "r1_grid_extension.py run 2-1-1 $s f64"; done
  for fn in xy hd; do for s in 0 1 2; do echo "r2_scaling.py kan $fn $s"; done; done
  for fn in exp xy hd; do for s in 0 1 2; do for d in 2 3 4; do echo "r2_scaling.py mlp $fn $s $d"; done; done; done
  echo "r3_symbolic.py 1e-3 1e-2"; echo "r4_continual.py"
} | run_par
$PY src/r1_grid_extension.py analyze; $PY src/r1_variants.py; $PY src/r2_scaling.py analyze; $PY src/r4_continual.py replot

# Hypotheses H1-H3 are not part of this script yet (planned for the final report).

# Stage 2: clean timing (run alone), demo
$PY src/r5_timing.py
$PY demo.py 2> /dev/null | tee logs/demo_output.txt
$PY src/render_terminal.py
echo "done; see RESULTS.md"
