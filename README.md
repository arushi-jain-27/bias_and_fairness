# FairAI

Run the simulations from this directory. They write result tables under `results/`. `regenerate_plots.py` reads three of those tables and writes the figures under `plots/`. None of the scripts take command-line arguments.

## Setup

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl
```

`results/` must already exist. `regenerate_plots.py` creates `plots/` if it is missing.

## Result tables

```bash
python main_regression.py
python ablation_study_regression.py
python main_classification.py
python main_real_data.py
```

`main_regression.py` and `ablation_study_regression.py` are the slow steps. The regression script fits a separation model for every protected group, including the multi-category settings. The ablation script repeats each configuration for 30 seeds. `main_real_data.py` downloads the UCI Adult data and the insurance charges table.

| Script | Table |
| --- | --- |
| `main_regression.py` | `results/regression_simulations.xlsx` |
| `ablation_study_regression.py` | `results/reg_ablation.xlsx` |
| `main_classification.py` | `results/classification_simulations.xlsx` |
| `main_real_data.py` | `results/real_classification_adult.xlsx`, `results/real_regression_insurance.xlsx` |

Each workbook is one table: one row per protected class, with the group size and fairness scores. The simulation tables also include the simulation setting. Regression rows also include the injected target and prediction multipliers and the DALEX separation score.

## Figures

`regenerate_plots.py` reads `regression_simulations.xlsx`, `reg_ablation.xlsx`, and `classification_simulations.xlsx`. Rerun it after any of those three tables changes.

```bash
python regenerate_plots.py
```

- `plots/target_pred.pdf`
- `plots/reg1-3.pdf`
- `plots/reg4-5.pdf`
- `plots/ablations_utup.pdf`, `plots/ablations_utbp.pdf`, `plots/ablations_btup.pdf`, `plots/ablations_btbp.pdf`
- `plots/class.pdf`
