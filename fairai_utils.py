import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


value_columns = ['Fairness_pred']

def weighted_avg(group):
    d = {}
    total_weight = group["count"].sum()
    for col in value_columns:
        d[f'weighted_{col}'] = (group[col] * group["count"]).sum() / total_weight
    d['count'] = total_weight  # Preserve the total count
    return pd.Series(d)



def calculate_model_fairness(df: pd.DataFrame, target_feature: str, prediction_col: str, protected_groups: list):
    """
    Calculate model fairness scores for multiple protected groups
    Args:
        - df (pd.DataFrame): Input dataframe with model features, target, predicted scores and protected features
        - protected_groups (list): List of protected group column names
        - prediction_col (str): column name of predicted scores
        - target_feature (str): column name of targets

    Returns:
        pd.DataFrame: A DataFrame containing fairness scores for all protected groups
    """



    df_scored = df.copy()

    df_count = len(df_scored)

    final_df_list = []
    
    for i, protected_feature in enumerate(protected_groups):
        fairness_df = calculate_model_fairness_classwise(df_scored, protected_feature, prediction_col)
        
        fairness_df = fairness_df.sort_values(['Protected_Class'])
        final_df_list.append(fairness_df)
    
    # Combine all fairness dataframes
    final_df = pd.concat(final_df_list, ignore_index=True)
    
    final_bnf_df = final_df[['Protected_Feature', 'Protected_Class', 'count', 'Fairness_pred']]

    return final_bnf_df



def calculate_model_fairness_classwise(df: pd.DataFrame, protected_group: str, prediction_col: str) -> pd.DataFrame:
    """
    Calculate model fairness scores for a specific protected group
    Args:
        - df (pd.DataFrame): Input dataframe with model features, target, predicted scores and protected features
        - protected_group (str): Any protected group column name
        - prediction_col (str): column name of predicted scores

    Returns:
        pd.DataFrame: A DataFrame containing fairness scores for a specific protected_group
    """
   
    # Group by protected group and calculate aggregates
    grouped_df = df.groupby(protected_group).agg({
        prediction_col: ['count', 'sum'],
    }).reset_index()
    
    # Flatten column names
    grouped_df.columns = [protected_group, 'count', 'predicted']
    
    # Calculate totals
    sum_counts = grouped_df['count'].sum()
    sum_preds = grouped_df['predicted'].sum()
    
    # Calculate fairness metrics
    grouped_df['average_pred'] = grouped_df['predicted'] / grouped_df['count']
    grouped_df['otheravg_pred'] = (sum_preds - grouped_df['predicted']) / (sum_counts - grouped_df['count'])
    
    # Calculate fairness ratios
    def calculate_fairness_ratio(avg_col, other_avg_col):
        ratio1 = avg_col / other_avg_col
        ratio2 = other_avg_col / avg_col
        return np.where(np.abs(ratio1) < np.abs(ratio2), ratio1, ratio2)
    
    # Calculate prediction fairness using ratio-based approach
    grouped_df['Fairness_pred'] = calculate_fairness_ratio(grouped_df['average_pred'], grouped_df['otheravg_pred'])
    
    
    # Rename and select columns
    final_df = grouped_df.rename(columns={protected_group: 'Protected_Class'})
    final_df['Protected_Feature'] = protected_group
    
    # Select final columns
    final_df = final_df[['Protected_Class', 'count', 'Fairness_pred', 'Protected_Feature']]
    
    return final_df


def calculate_regression_target_fairness(df: pd.DataFrame, target_feature: str, protected_groups: list):
    """
    Per-group target fairness for regression via point-biserial correlation (group vs rest).
    Returns one row per protected class with Fairness_target = 1 - |r_pb|.
    """
    rows = []
    y = df[target_feature].to_numpy()
    # population std (ddof=0) to keep it in [0,1]
    sigma = np.std(y)
    if sigma == 0 or np.isnan(sigma):
        sigma = 1e-12  # avoid divide by zero; fairness will be ~1

    for protected_feature in protected_groups:
        for g, grp_df in df.groupby(protected_feature):
            a = (df[protected_feature] == g).astype(int).to_numpy()
            p = a.mean()
            if p <= 0 or p >= 1:
                r_pb = 0.0  # degenerate group; treat as no association
            else:
                mu1 = y[a == 1].mean()
                mu0 = y[a == 0].mean()
                r_pb = ((mu1 - mu0) / sigma) * np.sqrt(p * (1 - p))

            rows.append({
                "Protected_Feature": protected_feature,
                "Protected_Class": g,
                "count": int(a.sum()),
                "Fairness_target": 1.0 - abs(float(r_pb)),
                # optional diagnostics:
                # "r_pb": float(r_pb), "p_group": float(p), "mu1": mu1, "mu0": mu0
            })

    return pd.DataFrame(rows)


def calculate_classification_target_fairness(df: pd.DataFrame, target_feature: str, protected_groups: list):
    """
    Target fairness for classification via per-group Cramér's V.
    For each protected group value g, build a 2 x C table [g vs rest] x [class],
    compute V_g, and set Fairness_target = 1 - V_g (higher = more fair).
    """


    def _cramers_v_2xC(table_2xC: np.ndarray) -> float:
        """
        Cramér's V for a 2 x C contingency table (no bias correction).
        V in [0,1]; for 2xC, V = sqrt(chi2 / n).
        """
        chi2, _, _, _ = chi2_contingency(table_2xC)
        n = table_2xC.sum()
        return 0.0 if n <= 0 else float(np.sqrt(chi2 / n))


    final_df_list = []

    for protected_feature in protected_groups:
        print(f"Computing classification target fairness (Cramér's V) for {protected_feature}")

        # protected x class contingency; keep its column order for alignment
        contingency = pd.crosstab(df[protected_feature], df[target_feature])
        class_order = list(contingency.columns)

        rows = []
        for group_name in contingency.index:
            counts_group = contingency.loc[group_name, class_order].values
            counts_rest  = contingency.drop(index=group_name).sum(axis=0).reindex(class_order).values
            table_2xC = np.vstack([counts_group, counts_rest])

            V_g = _cramers_v_2xC(table_2xC)
            fairness_score = 1.0 - V_g  # higher => closer to no association (fair)

            rows.append({
                "Protected_Feature": protected_feature,
                "Protected_Class": group_name,
                "count": int(counts_group.sum()),
                "Fairness_target": fairness_score,
            })

        final_df_list.append(pd.DataFrame(rows))

    return pd.concat(final_df_list, ignore_index=True)


    
def fair_ai_results(df, protected_groups, target_feature="target", prediction_cols=["prediction"], 
                   task_type="regression"):
    """
    Calculate fairness results across quantiles (regression) or classes (classification)
    Args:
        - df (pd.DataFrame): Input dataframe
        - target_feature (str): column name of target feature
        - protected_groups (list): List of protected group column names
        - prediction_cols (list): list of prediction column names
            * For regression: single column name, e.g., ["prediction"]
            * For classification: probability columns, e.g.,["prob_class_0", "prob_class_1", "prob_class_2"] for classification
        - task_type (str): "regression" or "classification" (default: "regression")
    
    Returns:
        pd.DataFrame: Weighted fairness results across quantiles (regression) or classes (classification)
    """
    
    df_copy = df.copy()
    
    pred_bucket_list = []
    
    if task_type == "regression":
        # Original quantile-based bucketing for regression
        if len(prediction_cols) != 1:
            raise ValueError("For regression, prediction_cols must contain exactly one column")

        bnf_target = calculate_regression_target_fairness(df_copy, target_feature, protected_groups)[[
            "Protected_Feature", "Protected_Class", "count", "Fairness_target"
        ]].rename(columns={"Fairness_target": "weighted_Fairness_target"})
        
        df_copy['quantile'] = pd.qcut(df_copy[target_feature], q=10, labels=False) + 1
        
        for i in range(1, 11):
            quantile_df = df_copy[df_copy['quantile'] == i].copy()
            q = calculate_model_fairness(quantile_df, target_feature, prediction_col=prediction_cols[0], protected_groups=protected_groups)
            q['bucket'] = f"q{i}"
            q['bucket_type'] = 'quantile'
            pred_bucket_list.append(q)

            
    elif task_type == "classification":
        # For classification, we need TWO types of analysis:
        # 1) Target fairness across protected groups (overall, via chi-square)
        # 2) Prediction fairness per true class bucket (class-based)

        unique_classes = sorted(df_copy[target_feature].unique())

        # Require one probability column per class
        if len(prediction_cols) != len(unique_classes):
            raise ValueError(f"For classification, prediction_cols must contain {len(unique_classes)} columns (one per class), got {len(prediction_cols)}")

        # --- Method 1: Target fairness (overall) ---
        # This returns one row per protected class with 'Fairness_target' and 'count'.
        # There is no bucket dimension to weight over, so we simply carry it through
        # as the final weighted target fairness per group.
        bnf_target = calculate_classification_target_fairness(df_copy, target_feature, protected_groups)[[
            "Protected_Feature", "Protected_Class", "count", "Fairness_target"
        ]].rename(columns={"Fairness_target": "weighted_Fairness_target"})


        # --- Method 2: Class-specific prediction fairness ---
        
        for class_label in unique_classes:
            # Create bucket for this class (all samples where true class = class_label)
            class_df = df_copy[df_copy[target_feature] == class_label].copy()

            # Find the probability column for this class
            prob_col = None
            for col in prediction_cols:
                if str(class_label) in col:  # e.g., "prob_class_0", "prob_0", etc.
                    prob_col = col
                    break
            if prob_col is None:
                raise ValueError(f"Could not find probability column for class {class_label} in {prediction_cols}")



            q = calculate_model_fairness(class_df, target_feature, prediction_col=prob_col, protected_groups=protected_groups)
            q['bucket'] = f"class_{class_label}"
            q['bucket_type'] = 'prediction_fairness'
            pred_bucket_list.append(q)




        # Merge so that final weighted_Fairness_target comes from target view
        # and weighted_Fairness_pred comes from prediction view only

    
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression' or 'classification'")


    pred_buckets_df = pd.concat(pred_bucket_list, ignore_index=True)
    bnf_pred = pred_buckets_df.groupby(["Protected_Feature", "Protected_Class"]).apply(weighted_avg).reset_index()
    bnf = bnf_target[["Protected_Feature", "Protected_Class", "weighted_Fairness_target"]].merge(
        bnf_pred,
        on=["Protected_Feature", "Protected_Class"],
        how="inner"
    )

    return bnf

 