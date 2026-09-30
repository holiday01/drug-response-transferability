#!/usr/bin/env bash
# Regenerate every result, figure and supplementary table in the paper.
# Requires the input data under data/ (see README.md). Order matters: later
# steps read results/per_drug_external.csv written by the first step.
set -euo pipefail
cd "$(dirname "$0")"
# Heavy scikit-learn jobs are unstable with unrestricted BLAS threading.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

for s in run_external_validation run_transferability_analysis run_consensus \
         run_screen_validation run_robustness run_tcga_bridge run_three_assay \
         run_stats_rigor run_sensitivity run_biology \
         make_figures make_figures2 make_supp_figures build_supp verify_numbers; do
  echo "=== $s"
  python scripts/$s.py
done
