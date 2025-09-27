import pandas as pd
from fairai_utils import fair_ai_results
from real_data_utils import load_adult_classification, load_insurance_regression


def run_classification_adult():
    df = load_adult_classification(sample_frac=1.0)
    protected_groups = ['sex', 'race', 'occupation']
    # 2 classes -> provide 2 prob columns
    df_fairai = fair_ai_results(
        df,
        protected_groups=protected_groups,
        prediction_cols=['prob_class_0', 'prob_class_1'],
        task_type='classification'
    )
    df_fairai['dataset'] = 'Adult'
    return df_fairai


def run_regression_insurance():
    df = load_insurance_regression(sample_frac=1.0)
    # choose protected columns present in df
    protected_groups = [c for c in df.columns if c not in ['target', 'prediction']]
    df_fairai = fair_ai_results(
        df,
        protected_groups=protected_groups,
        prediction_cols=['prediction'],
        task_type='regression'
    )
    df_fairai['dataset'] = 'Insurance'
    return df_fairai


if __name__ == "__main__":
    # Classification: Adult
    adult_results = run_classification_adult()
    adult_results.to_excel('results/real_classification_adult.xlsx', index=False)

    # Regression: Insurance Charges
    insurance_results = run_regression_insurance()
    insurance_results.to_excel('results/real_regression_insurance.xlsx', index=False)

    print("Saved results to results/real_classification_adult.xlsx and results/real_regression_insurance.xlsx")


