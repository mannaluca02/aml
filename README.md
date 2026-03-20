# AML - Product Affinity Model

FHNW Mini-Challenge for the module "Applied Machine Learning" (Angewandte Data Science).
Binary classification model to predict which customers of the fictional "Czech Fin Banka" are most likely to accept a credit card, in order to optimize a targeted marketing campaign.

## Project Structure

```
aml/
├── notebooks/          # Analysis notebooks (executed in order)
├── src/                # Reusable Python modules
├── data/
│   ├── raw/            # Original CSV files (semicolon-delimited)
│   └── processed/      # Intermediate outputs (parquet)
├── erd/                # Entity-Relationship Diagram artifacts
├── requirements.txt
└── README.md
```

### `notebooks/`

| Notebook | Phase | Description |
|---|---|---|
| `00_eda.ipynb` | 1 — Data Understanding & Study Design | PK/FK validation, cardinality checks, ERD, data quality analysis, exploration of all tables, study design with tenure-matched pseudo-event assignment, stratified train/test split. Outputs `study_table.parquet`. |
| `01_feature_engineering.ipynb` | 2 — Feature Engineering | Transaction rollup filtering, tsfresh automatic feature extraction, baseline features (5 required), static/contextual features, feature matrix assembly, feature selection. Outputs `feature_matrix.parquet`. |
| `02_modeling.ipynb` | 3 — Predictive Modeling & Validation | Baseline LogReg (5 features), extended LogReg (22 features), RandomForest, GradientBoosting, HistGradientBoosting with hyperparameter tuning. ROC/PR/Lift curves, Top-N% lists, feature importance, confusion matrices. W&B experiment tracking. |

Run notebooks in order — each depends on outputs from the previous phase.

### `src/`

| Module | Description |
|---|---|
| `data.py` | Data loading and cleaning: `load_raw_data()` loads all 8 CSVs with date parsing, Czech string decoding, gender extraction. `load_study_table()`. Shared constants (`RANDOM_STATE`, `ROLLUP_MONTHS`, paths). |
| `features.py` | Feature engineering: `filter_transactions_to_rollup()` for temporal filtering, `compute_tsfresh_features()` for automatic time series features, `compute_baseline_features()` for the 5 assignment-required features, `compute_static_features()` for district/loan/order/account features, `build_feature_matrix()` master builder. |
| `modeling.py` | Modeling utilities: `build_pipeline()` for leakage-free sklearn pipelines, `evaluate_on_test()` for test-set metrics, `plot_roc_curves()`/`plot_pr_curves()`/`plot_lift_curve()` for evaluation charts, `compute_top_n_list()`/`compare_top_n_lists()` for customer prioritization, `plot_feature_importance()` for interpretability. |

### `data/`

**`data/raw/`** — 8 source tables from the PKDD'99 Discovery Challenge (Czech bank financial dataset):

| File | Records | Description |
|---|---|---|
| `account.csv` | 4,500 | Bank accounts |
| `card.csv` | 892 | Credit cards (linked via disposition) |
| `client.csv` | 5,369 | Customers |
| `disp.csv` | 5,369 | Dispositions (bridge between client and account) |
| `district.csv` | 77 | Regional demographics |
| `loan.csv` | 682 | Loans |
| `order.csv` | 6,471 | Permanent orders |
| `trans.csv` | 1,056,320 | Transactions |

**`data/processed/`** — generated outputs (not committed, reproducible from notebooks):

| File | Description |
|---|---|
| `study_table.parquet` | One row per account with event date, rollup window, target label, and train/test split |
| `feature_matrix.parquet` | One row per account with all predictive features, target, and split |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11+.
