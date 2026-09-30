"""Central path configuration.

Every analysis script imports its input and output locations from here, so the
pipeline runs on any machine once the public input data are placed under
DATA_DIR (see README.md). Each location can be overridden by an environment
variable.
"""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("DRT_DATA_DIR", REPO_DIR / "data"))
# DepMap / CCLE release files (expression, CTRP AUC, GDSC2 replicate-level dose, Model.csv, Gene.csv)
CCLE_DIR = Path(os.environ.get("DRT_CCLE_DIR", DATA_DIR / "depmap"))
# PRISM Repurposing secondary screen, dose-response curve parameters
PRISM_AUC = Path(os.environ.get("DRT_PRISM_AUC", DATA_DIR / "prism" / "PRISM_secondary_AUC.csv"))
# Drug annotation table (drug, broad_id, moa, target, smiles, phase) from the Drug Repurposing Hub
DRUG_CATALOG = Path(os.environ.get("DRT_DRUG_CATALOG", DATA_DIR / "repurposing_hub" / "drug_catalog.parquet"))
# TCGA protein-coding log-TPM expression and patient drug-response table
TCGA_EXPR = Path(os.environ.get("DRT_TCGA_EXPR", DATA_DIR / "tcga" / "TCGA_RNAseq_logTPM_protein_coding.csv"))
TCGA_DRUG = Path(os.environ.get("DRT_TCGA_DRUG", DATA_DIR / "tcga" / "tcga_paire.csv"))
# GDSC2 release 8.5 fitted dose-response table (official fitted-curve AUC), from cancerrxgene.org;
# SHA-256 in checksums/SHA256SUMS. Used by the reference/test (revision) analyses.
GDSC2_FITTED = Path(os.environ.get("DRT_GDSC2_FITTED",
                                   DATA_DIR / "gdsc" / "GDSC2_fitted_dose_response_27Oct23.xlsx"))
# Manuscript source checked by verify_numbers_rev.py (optional; not distributed).
MANUSCRIPT_TEX = Path(os.environ.get("DRT_MANUSCRIPT_TEX", REPO_DIR / "manuscript" / "main.tex"))

RESULTS_DIR = Path(os.environ.get("DRT_RESULTS_DIR", REPO_DIR / "results"))
OUTPUT_DIR = Path(os.environ.get("DRT_OUTPUT_DIR", REPO_DIR / "outputs"))
FIG_DIR = OUTPUT_DIR / "figures"
SUPP_FIG_DIR = OUTPUT_DIR / "supp_figures"
FIG_TIFF_DIR = OUTPUT_DIR / "figures_tiff"
# Result folders of the reference/test (revision) analyses
REV1_DIR = RESULTS_DIR / "rev1_independent"
REV2_DIR = RESULTS_DIR / "rev2_three_assay"
REV3_DIR = RESULTS_DIR / "rev3_consensus"

for _d in (RESULTS_DIR, FIG_DIR, SUPP_FIG_DIR, FIG_TIFF_DIR):
    _d.mkdir(parents=True, exist_ok=True)
