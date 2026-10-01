#!/usr/bin/env bash
# Regenerate every result and figure in the paper and check its headline numbers.
# Requires the input data under data/ (see README.md). Order matters:
#  * run_external_validation writes results/per_drug_external.csv (drug panel
#    for REV-1 and input to most later steps);
#  * run_consensus writes R3_consensus_per_drug.csv (drug list for REV-3);
#  * run_robustness writes R6_bidirectional_per_drug.csv (Fig. 5D);
#  * the *_rev figure scripts run after the original ones and
#    overwrite Figs 1, 2, 4, 5, 6 and S1, S2, S9 with the final versions
#    (make_figures2 writes obsolete Figs 5 and 6 and must run before them).
#  Figure files keep their working names; in the paper fig5_robustness is
#  Figure 3 and fig3_determinants is Figure 5.
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
         run_three_assay_rev run_consensus_rev analyze_rev_extra \
         run_refsize_single_draw_rev run_three_assay_blockperm_rev \
         analyze_consensus_min_splits_rev; do
  echo "=== $s"
  python scripts/$s.py
done

# Figures and number checks
for s in make_figures make_figures2 make_supp_figures \
         make_figures_rev make_fig6_rev make_supp_figures_rev \
         verify_numbers verify_numbers_rev; do
  echo "=== $s"
  python scripts/$s.py
done
