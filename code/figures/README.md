# Figure scripts

Submission files are named `submission/Figure_1.pdf` to `Figure_7.pdf`.

| Figure | Script | Data |
|---|---|---|
| Fig. 1 architecture | `fig1_pipeline/make_fig1_tikz.py` (TikZ, then `pdflatex figures/fig_pipeline.tex`) | `outputs/fig1_pipeline_assets_20261105/case_5_3_225/` |
| Fig. 2 datasets and input conditions | `fig_datasets.py` | `outputs/figure_data_20261105/` (object crops, scenes, per-condition CSVs), case npz above |
| Fig. 3 pose returned by SARef | `fig_selection.py` | `outputs/figure_data_20261105/per_condition_*.csv` |
| Fig. 4 risk–coverage | `fig_risk_coverage.py` | `outputs/p0_risk_coverage_20261104/` |
| Fig. 5 qualitative cases | `fig_qualitative.py` | `outputs/a_main_method_qualitative_20261105/` |
| Fig. 6 per-object AR change | `fig_per_object.py` | `outputs/figure_data_20261105/per_condition_*.csv` |
| Fig. 7 AR change vs initial error | `fig_gain_vs_error.py` | `outputs/figure_data_20261105/per_condition_*.csv` |

`figstyle.py` holds the shared style: Liberation Sans (Arial metrics), 7 pt lettering, and Elsevier widths of 90 and 190 mm. Paths are set by environment variables:
- `FIGDATA`: the folder with the unzipped `figure_data_20261105` content;
- `FIGOUT`: the output folder;
- `RCDATA`: the risk-coverage folder;
- `QDATA` and `QIMG`: the qualitative cases and their RGB images;
- `FIGCASE`: the case npz.

`outputs/figure_data_20261105/` was produced by `export_figure_data.py` and `make_object_crops.py` in the same folder. Both are read-only re-aggregations of the frozen per-condition outcomes and of the BOP test images.

Each script also writes a `*_values.json` with the plotted numbers. These are the numbers quoted in the text.
