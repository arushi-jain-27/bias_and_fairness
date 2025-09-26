import pandas as pd
import numpy as np

# Minimal, dependency-light dataset loaders that yield dataframes compatible with fair_ai_results
# We avoid heavy libs (aif360, folktables) to keep this repo self-contained.


def load_adult_classification(sample_frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    Load the Adult Census Income dataset from UCI via OpenML mirror.
    Returns a DataFrame with:
        - protected attributes: 'sex', 'race'
        - features (not strictly needed by fairness metrics here)
        - target: 0/1 (income >50K)
        - prob_class_0, prob_class_1: simple logistic baseline probabilities
    Note: Uses a lightweight heuristic model (logit on few features) for demo.
    """
    # Load directly from the UCI repository (no header in file)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(url, header=None, names=columns, na_values=['?'], skipinitialspace=True)

    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Expected columns
    # income is ">50K" or "<=50K"; protected: sex, race
    target_col = 'income'
    protected_groups = ['sex', 'race']

    # Clean unknowns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=protected_groups + [target_col])

    # Map target to 0/1
    df['target'] = (df[target_col] == '>50K').astype(int)

    # Simple numeric features for a quick baseline score
    # Create a few engineered features from known columns if present
    def safe_numeric(col):
        return pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(np.nan, index=df.index)

    age = safe_numeric('age')
    hours = safe_numeric('hours-per-week') if 'hours-per-week' in df.columns else safe_numeric('hours_per_week')
    capital_gain = safe_numeric('capital-gain') if 'capital-gain' in df.columns else safe_numeric('capital_gain')
    capital_loss = safe_numeric('capital-loss') if 'capital-loss' in df.columns else safe_numeric('capital_loss')

    # Basic linear score
    score = (
        0.04 * (age.fillna(age.median()))
        + 0.06 * (hours.fillna(hours.median()))
        + 0.0005 * (capital_gain.fillna(0))
        - 0.0005 * (capital_loss.fillna(0))
    )
    # Clip and pass through sigmoid to form a probability
    score = np.clip(score, -10, 10)
    prob1 = 1 / (1 + np.exp(-score))
    prob0 = 1 - prob1

    df['prob_class_0'] = prob0
    df['prob_class_1'] = prob1

    # Optional downsample for speed
    if 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)

    # Keep only columns we need for fairness evaluation
    keep_cols = protected_groups + ['target', 'prob_class_0', 'prob_class_1']
    return df[keep_cols].reset_index(drop=True)



def load_insurance_regression(sample_frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    Load the Insurance Charges dataset with protected attributes and a simple regression baseline.
    Source: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
    Produces:
        - protected attributes: 'sex', 'smoker', 'region'
        - target: continuous 'charges'
        - prediction: baseline ridge-like linear score using one-hot features
    """
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)

    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    if 'charges' not in df.columns:
        raise ValueError("Expected column 'charges' not found in Insurance dataset")

    protected_groups = [c for c in ['sex', 'smoker', 'region'] if c in df.columns]

    # Build simple linear baseline with one-hot encoding for categorical features
    categorical_cols = [c for c in df.columns if df[c].dtype == object and c != 'charges']
    numeric_cols = [c for c in df.columns if df[c].dtype != object and c != 'charges']

    X_cats = pd.get_dummies(df[categorical_cols], drop_first=True) if len(categorical_cols) else pd.DataFrame(index=df.index)
    X_nums = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(df[numeric_cols].median()) if len(numeric_cols) else pd.DataFrame(index=df.index)
    X = pd.concat([X_nums, X_cats], axis=1)
    # Ensure numeric dtype throughout
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y = df['charges']

    # Closed-form ridge solution for stability
    X_design = np.column_stack([np.ones(len(X)), X.values.astype(float)])
    lam = 1e-3
    XtX = X_design.T @ X_design + lam * np.eye(X_design.shape[1])
    Xty = X_design.T @ y.values
    coef = np.linalg.solve(XtX, Xty)
    pred = X_design @ coef

    df['target'] = y
    df['prediction'] = pred

    if 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)

    keep_cols = protected_groups + ['target', 'prediction']
    return df[keep_cols].reset_index(drop=True)

def get_protected_groups_from_df(df: pd.DataFrame, exclude: list) -> list:
    """
    Return column names that are protected attributes in df, excluding any listed.
    For our loaders we pass the known protected columns explicitly.
    """
    return [c for c in df.columns if c not in exclude]


