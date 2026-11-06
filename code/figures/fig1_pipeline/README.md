# Fig. 1 (pipeline) — reproducible build

The figure (architecture layout) shows one real target: unseen T-LESS object 5, scene 3, image 225, mild perturbation, 30% local depth outliers.
All panel data come from `outputs/fig1_pipeline_assets_20261105/case_5_3_225/`:
- `case_observation.npz` holds the observation replay. It was written by `source/extract_case.py`, and the depth, patch, mask and K SHA-256 checks pass.
- `candidate_predictions.csv` holds the frozen-model outputs for this target.

Build steps (Python with numpy, pandas, scipy and matplotlib; LaTeX with TikZ):

    cd figure_scripts/fig1_pipeline
    python make_assets_1.py      # images, depth, CAD view, candidate zoom, residual / silhouette maps, output overlay
    python make_assets_2.py      # point clouds (P2P / filtered), NN-distance map
    python make_fig1_tikz.py     # architecture layout; writes ../../figures/fig_pipeline.tex
    cd ../../figures && pdflatex fig_pipeline.tex

To use a different case folder, set the environment variable `SAREF_FIG1_CASE`.
The cloud panel marks points as removed with the frozen keep rule `reliability > 0.35`.
The canvas is 17.5 cm wide, so its 7 pt labels stay at 7 pt in a two-column journal layout.
The feature-family bar heights come from grouping the 191 names in feature_schema.json (decoupled_development_20261020_r4) by name prefix. The groups are bank registration scores 60, rendered depth 36, silhouette/boundary 32, nearest neighbour 28, target context 18, motion/flags 10 and joint support change 7.
