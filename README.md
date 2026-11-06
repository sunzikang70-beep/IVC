# SARef: selective acceptance of local refinements for RGB-D 6D object pose estimation

This repository holds the code, the frozen model and the per-condition results behind the manuscript *SARef: Selective acceptance of local refinements for RGB-D 6D object pose estimation*, submitted to *Image and Vision Computing*.

SARef builds six bounded candidate updates for an input pose: point-to-point (P2P) and reliability-filtered P2P registration, each with motion caps of 1/4, 1/2 and 1. Gradient-boosted models, frozen after development on 22 T-LESS objects, predict how each candidate changes BOP recall (VSD, MSSD, MSPD) and ADD-S. The candidate with the largest predicted utility is accepted only if it satisfies the support, risk and per-metric constraints; otherwise the input pose is kept.

## Layout

| Folder | Content |
|---|---|
| `code/pipeline/<run>/` | Exact source snapshot recorded by each run (candidate generation, observable features, labels, training, frozen inference). File-name prefixes are those of the snapshot. |
| `code/analysis/` | Scripts that turn the result tables into the numbers in the paper (object-equal means, 20,000-draw object-cluster bootstrap, seed 20261104). |
| `code/figures/` | Scripts for Figs. 1–7 (see `code/figures/README.md`). |
| `models/` | Frozen model `development_candidate.joblib` (SHA-256 `a3836d3bfbb0dda4927578fdfb53389830fe0cff2bdc439f46de8cdcb2a5917c`), the 191-feature schema and the frozen policy receipt. |
| `results/tless_development/` | Development bank (22 objects): out-of-object predictions, policy grid, labels and labelled candidates. |
| `results/tless_unseen/` | Frozen runs on the 8 unseen T-LESS objects (`unseen5`: objects 5, 13, 18, 21, 27; `unseen3`: objects 1, 8, 22). |
| `results/ycbv_external/` | Frozen external run on 21 YCB-V objects (24,144 conditions) with the locked protocol and input hashes. |
| `results/tables_saref/` | Paper tables with SARef as the main method: `summary.csv`, `paired_bootstrap.csv`, `oracle_headroom.csv`, `worst_regime.csv`, `rule_ablation_A.csv`, `adds_A.csv`. |
| `results/tless_tables/` | Inputs and outputs of the main table, the risk–coverage curves, the rule replay, the residual-correction ablation, the matched-coverage comparison, the observability gate, the decision-model check and the qualitative-case selection. |
| `results/runtime/` | Per-stage timings and environment of the runtime benchmark. |
| `results/figure_data/` | Per-condition tables used for Figs. 2, 3, 6 and 7. |
| `MANIFEST_SHA256.csv` | Size and SHA-256 of every file. |

Large intermediate tables are not in the git tree; they are listed in `EXTRA_FILES.md`. They are the YCB-V features, labels and scores, and the observable features of the T-LESS banks.

## Names used in the data files

| In the files | In the paper |
|---|---|
| `A ordinary ranking`, `A_delta_rank` | SARef |
| `B fixed residual`, `C state residual`, `D joint (ours)` | Ablation variants B, C and D (residual correction). D was the designated method before the final analysis, and its label is kept for provenance. |
| `Initial`, `protocolB_e2e_initial_pose` | Input pose $T_0$ |
| `P2P s0.25` … `Filtered P2P s1.00`, `protocolF_se3_trust_{p2p,filtered}_s{025,050,100}` | Fixed refinements / candidates (P2P or filtered P2P, cap 0.25, 0.5, 1) |
| `init_regime`: `native`, `mild`, `hard` | Initialization regimes |
| `input_depth_condition`: `native_depth`, `local_outlier_30pct` | Depth conditions |

## Where the numbers come from

| Paper item | Result files | Script |
|---|---|---|
| Tables 2, 3, 5, 6 and Sections 5–7 | `results/tables_saref/` | `code/analysis/analyze_A_main_method.py` |
| Table 4 (rule ablation) | `results/tables_saref/rule_ablation_A.csv`, `results/tless_tables/rule_replay/` | `code/analysis/analyze_A_main_method.py` |
| ADD-S analysis (Section 5.5) | `results/tables_saref/adds_A.csv` | same |
| Residual-correction ablation | `results/tless_tables/residual_ablation/`, `results/tless_tables/matched_coverage/` | `code/pipeline/same_bank_residual_ablation/`, `code/pipeline/matched_risk_coverage/` |
| Observability gate and decision-model check | `results/tless_tables/observability_gate/`, `results/tless_tables/decision_model_check/` | `code/analysis/extract_observability_state_features.py`, `observability_residual_gate.py`, `decision_model_check/` |
| Table 7 (runtime) | `results/runtime/` | `code/analysis/run_p0_runtime_benchmark.py` |
| Fig. 4 (risk–coverage) | `results/tless_tables/risk_coverage/` | `code/analysis/risk_coverage/`, `code/figures/fig_risk_coverage.py` |
| Fig. 5 (qualitative cases) | `results/tless_tables/qualitative_cases/` | `code/analysis/qualitative_cases/`, `code/figures/fig_qualitative.py` |
| Figs. 2, 3, 6, 7 | `results/figure_data/` | `code/figures/` |

The analysis scripts read the original workspace layout (`outputs/<run folder>/...`). The run folders map to this repository as follows:

| Workspace folder | Repository folder |
|---|---|
| `decoupled_development_20261020_r4` | `results/tless_development` |
| `decoupled_frozen_unseen5_20261021_r4`, `decoupled_frozen_unseen3_20261023_r1` | `results/tless_unseen/unseen5`, `unseen3` |
| `decoupled_frozen_ycbv_external_20261104` | `results/ycbv_external` |
| `p0_main_table_baselines_20261104` | `results/tless_tables/main_table` |
| `p0_risk_coverage_20261104` | `results/tless_tables/risk_coverage` |
| `joint_optimization_review_20261026_r2` | `results/tless_tables/rule_replay` |
| `same_bank_residual_ablation_20261026_r1` | `results/tless_tables/residual_ablation` |
| `matched_risk_coverage_20261026_r1` | `results/tless_tables/matched_coverage` |
| `a_main_method_tables_20261105` | `results/tables_saref` |

## Data not included

The T-LESS and YCB-V images, models and ground truth come from the BOP benchmark (https://bop.felk.cvut.cz/datasets/). The external inputs come from the public BOP submissions: CNOS/FastSAM masks, MegaPose poses (T-LESS) and CosyPose single-view poses (YCB-V). Neither the datasets nor these inputs are redistributed here. The object crops in Fig. 2 are regenerated from the BOP test images by `code/figures/data_export/make_object_crops.py`.

## Environment

Experiments ran on Windows 11 with Python 3.10, scikit-learn 1.7.2, Open3D 0.19.0, NumPy and pandas. BOP errors were computed with the BOP toolkit, and runtime was measured on an Intel Core i5-11400H CPU with 12 threads.

## Licence

Code: MIT (`LICENSE`). Data, model and result tables: CC BY 4.0 (`LICENSE-DATA`).

## Citation

[Citation to be added after publication.]
