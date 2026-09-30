#!/usr/bin/env bash
# Regenerate every result, figure and supplementary table in the paper.
# Requires the input data under data/ (see README.md). Order matters:
#  * run_external_validation writes results/per_drug_external.csv (drug panel
#    for REV-1 and input to most later steps);
#  * run_consensus writes R3_consensus_per_drug.csv (drug list for REV-3);
#  * run_robustness writes R6_bidirectional_per_drug.csv (Fig. 5D);
#  * the *_rev figure/supplement scripts run after the original ones and
#    overwrite Figs 1, 2, 4, 5 and S1, S2, S9 with the primary-design versions.
set -euo pipefail
cd "$(dirname "$0")"
# Heavy scikit-learn jobs are unstable with unrestricted BLAS threading.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

# Supporting analyses (original single-split design) and shared inputs
for s in run_external_validation run_transferability_analysis run_consensus \
         run_screen_validation run_robustness run_tcga_bridge run_three_assay \
         run_stats_rigor run_sensitivity run_biology; do
  echo "=== $s"
  python scripts/$s.py
done

# Primary analyses (reference/test design, GDSC2 fitted AUC)
for s in run_independent_validation analyze_independent_validation \
         run_three_assay_rev run_consensus_rev analyze_rev_extra; do
  echo "=== $s"
  python scripts/$s.py
done

# Figures, supplement and number checks
for s in make_figures make_figures2 make_supp_figures \
         make_figures_rev make_supp_figures_rev build_supp_rev \
         verify_numbers verify_numbers_rev; do
  echo "=== $s"
  python scripts/$s.py
done
