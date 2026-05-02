# Beyond Accuracy: Explanation Instability Under Decision Uncertainty in Credit Risk Classification

Code and experiments for the paper *Beyond Accuracy: Explanation Instability Under Decision Uncertainty in Credit Risk Classification* (Aldowaihi & Al-Jamimi).

This repository evaluates the reliability of post-hoc explanation methods (TreeSHAP and LIME) for credit-default ensemble classifiers, conditioned on the model's decision confidence. The headline contribution is a three-stratum analysis — **Confident-Correct**, **Borderline**, and **Confident-Incorrect** — that treats high-confidence-but-wrong predictions as a distinct unit of analysis rather than aggregating them into a global error pool.

## Headline findings

- Within-method stability follows a **non-monotonic U-shape** across the three strata: Borderline attributions are substantially less stable than either confident stratum, while Confident-Incorrect attributions are statistically indistinguishable from Confident-Correct ones.
- TreeSHAP and LIME **anti-correlate** at the Borderline stratum (median Spearman ρ ≈ −0.4 under both ensembles).
- Under Gradient Boosting, cross-method Top-5 feature overlap **collapses on Confident-Incorrect cases** (a degradation that does not appear under Random Forest), suggesting cross-explainer disagreement may be a more useful audit signal than within-method instability.

See the paper for the full discussion.

## Repository contents

```
.
├── Credit_XAI_Dalal_ALdowaihi.ipynb   # End-to-end experimental notebook
├── outputs/
│   ├── tables/                        # CSVs of all per-instance and summary results
│   ├── figures/                       # PNGs for every figure in the paper
│   └── models/                        # (optional) pickled trained models
├── data/
│   └── default_of_credit_card_clients.xls   # Auto-downloaded on first run
└── README.md
```

The notebook is the single source of truth — every table and figure in the paper is produced by it.

## Dataset

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) (Yeh & Lien, 2009). 30,000 instances, 23 features, ≈22% default rate. The notebook downloads the dataset automatically on first run.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `scikit-learn` 1.5.x
- `shap`
- `lime`
- `lightgbm` (preferred) or scikit-learn's `HistGradientBoostingClassifier` as fallback
- `statsmodels` (for Holm–Bonferroni correction)

Install everything in one line:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn shap lime lightgbm statsmodels
```

The notebook also runs in Google Colab without local setup — the first cell installs the non-default packages.

## Quick start

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
jupyter notebook Credit_XAI_Dalal_ALdowaihi.ipynb
```

Then **Kernel → Restart & Run All**. All outputs (tables and figures) are written to `outputs/` and embedded inline in the notebook.

For a fast smoke test that takes ~3–5 minutes instead of the full ~45–90 minutes, set `QUICK_MODE = True` in the configuration cell. Quick mode reduces samples per stratum from 100 to 20, perturbations per instance from 30 to 10, and LIME samples from 5000 to 1000. Quick-mode results are not the values reported in the paper.

## Reproducing the paper

All randomness is controlled by a single seed (`SEED = 42` in the configuration cell). Running the notebook end-to-end on a clean kernel reproduces every number in the paper, including:

- Test-set AUC-ROC (Random Forest 0.7787, Gradient Boosting 0.7778)
- Stratum populations under both ensembles (Table 8 of the paper)
- Per-stratum stability medians (Table 10)
- Mann–Whitney pairwise comparisons with Holm–Bonferroni correction (Tables 11 and 13)
- σ-sweep robustness check at σ ∈ {0.01, 0.03, 0.05} (Table 12)
- Cross-method agreement medians and distributions (Section 4.5–4.7)

> **Important:** Re-run the notebook with a clean kernel — do not rely on cached cell outputs. Some intermediate results (Experiment B and the final statistical tables) consume variables produced earlier in the notebook, and partial re-runs can produce stale numbers.

Expected runtime on a modern laptop:

| Stage | Time |
|---|---|
| Data loading + EDA | < 1 min |
| Train RF + GBM (with grid search) | 2–5 min |
| TreeSHAP / LIME explainers + global beeswarm | 1–2 min |
| Experiment A (perturbation stability, both models) | 20–40 min |
| σ-sweep on Random Forest | 10–20 min |
| Experiment B (cross-method agreement) | 5–10 min |
| Statistical tests + figures | < 1 min |

LIME is the dominant cost; reducing `LIME_NUM_SAMPLES` shortens the run at the cost of higher per-instance LIME variance.

## Method summary

- **Models:** Random Forest (bagging, `n_estimators=500`) and LightGBM (boosting, `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`), both trained with class-balanced weighting on a 70/15/15 stratified split.
- **Stratification:** test predictions partitioned by predicted probability `p̂` of default into Confident-Correct (`p̂ ≤ 0.20` or `p̂ ≥ 0.80`, correct), Borderline (`0.40 ≤ p̂ ≤ 0.60`), and Confident-Incorrect (confident but wrong). Up to 100 instances per stratum are sampled for the in-depth experiments.
- **Type-aware perturbation:** continuous features get additive Gaussian noise at σ × per-feature std; ordinal repayment-delay features get ±1 shifts with probability 0.20; categorical features are held fixed. A `|Δp̂| < 0.05` filter is applied to retained neighbors so that explanation drift is isolated from prediction drift.
- **Stability metrics (per instance):** Spearman ρ over the full attribution vector; Jaccard similarity over the Top-5 features by `|ϕ|`. Higher = more stable.
- **Statistical tests:** Mann–Whitney U (two-sided) of each stratum vs. Confident-Correct, with Holm–Bonferroni correction at m = 2 comparisons per metric. Effect size reported as rank-biserial correlation.

See `Credit_XAI_Dalal_ALdowaihi.ipynb` Section 3 (or the paper's Section 3) for the full protocol.

## Configuration knobs

All experimental parameters are in a single configuration cell at the top of the notebook:

| Parameter | Default | Meaning |
|---|---|---|
| `SEED` | 42 | Master random seed |
| `INSTANCES_PER_STRATUM` | 100 | In-depth analysis sample per stratum |
| `N_PERTURBATIONS` | 30 | Type-aware neighbors per instance |
| `PERTURB_SIGMA_HEADLINE` | 0.03 | Headline perturbation magnitude |
| `PERTURB_SIGMA_GRID` | (0.01, 0.03, 0.05) | σ-sweep robustness check |
| `PRED_STABILITY_FILTER` | 0.05 | `|Δp̂|` filter for retained neighbors |
| `LIME_NUM_SAMPLES` | 5000 | LIME local-neighborhood sample size |
| `CONFIDENT_LOW`, `CONFIDENT_HIGH` | 0.20, 0.80 | Confident-band thresholds |
| `BORDERLINE_LOW`, `BORDERLINE_HIGH` | 0.40, 0.60 | Borderline-band thresholds |
| `ALPHA` | 0.05 | Significance level |

## Output files

After a clean run, `outputs/tables/` contains:

- `test_metrics.csv` — predictive performance for both ensembles
- `stratum_counts.csv` — realized stratum populations
- `expA_random_forest_per_instance.csv`, `expA_gradient_boosting_per_instance.csv` — per-instance Experiment A results
- `expA_random_forest_sigma{0.01,0.05}_per_instance.csv` — σ-sweep results
- `expB_random_forest_per_instance.csv`, `expB_gradient_boosting_per_instance.csv` — per-instance Experiment B results
- `stat_tests_all.csv` — Mann–Whitney + Holm–Bonferroni test results for every metric × stratum × model
- `sigma_sweep_summary.csv` — per-stratum medians at each σ
- `global_importance.csv` — top-N feature importance for both models

`outputs/figures/` contains the corresponding PNGs (`expA_*`, `expB_heatmap_*`, `summary_expA_*`, `shap_beeswarm_*`, `global_importance.png`, etc.).

## Citation

If you use this code or build on the analysis, please cite the paper:

```bibtex
@misc{aldowaihi_aljamimi_2025_explanation_instability,
  title  = {Beyond Accuracy: Explanation Instability Under Decision
            Uncertainty in Credit Risk Classification},
  author = {Aldowaihi, Dalal and Al-Jamimi, Hamdi},
  year   = {2025},
  note   = {Computer Science Department, KFUPM}
}
```

The dataset itself should be cited as Yeh & Lien (2009).

## License

Code is released under the MIT License (see `LICENSE`). The UCI Default of Credit Card Clients dataset is distributed by the UCI Machine Learning Repository and subject to its terms.

## Contact

For questions about the methodology or results, please open an issue on this repository or contact the authors:

- Dalal Aldowaihi — `202514270@kfupm.edu.sa`
- Dr. Hamdi Al-Jamimi — `aljamimi@kfupm.edu.sa`

Computer Science Department, King Fahd University of Petroleum and Minerals (KFUPM), Dhahran, Saudi Arabia.
