import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

# Minimal, dependency-light dataset loaders that yield dataframes compatible with fair_ai_results
# We avoid heavy libs (aif360, folktables) to keep this repo self-contained.


def load_adult_classification(sample_frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    Load the Adult Census Income dataset from UCI via OpenML mirror.
    Returns a DataFrame with:
        - protected attributes: 'sex', 'race', 'occupation'
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
    df = pd.read_csv(url, header=None, names=columns, skipinitialspace=True)

    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Expected columns
    # income is ">50K" or "<=50K"; protected: sex, race
    target_col = 'income'
    protected_groups = ['sex', 'race', 'occupation']

    # Clean unknowns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    
    
    # Only drop rows with actual NaN values, not '?' strings
    df = df.dropna(subset=protected_groups + [target_col])

    # Map target to 0/1
    df['target'] = (df[target_col] == '>50K').astype(int)

    # Prepare features for scikit-learn logistic regression
    features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    # Create feature matrix
    X = df[features].fillna(df[features].median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression model
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr.fit(X_scaled, df['target'])
    
    # Get probabilities
    proba = lr.predict_proba(X_scaled)
    df['prob_class_0'] = proba[:, 0]
    df['prob_class_1'] = proba[:, 1]

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

    # Build features for scikit-learn linear regression
    numeric_cols = [c for c in df.columns if df[c].dtype != object and c != 'charges']
    
    # Prepare features
    X = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(df[numeric_cols].median())
    y = df['charges']

    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X, y)
    pred = lr.predict(X)

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


