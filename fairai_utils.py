import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


value_columns = ['average_target', 'otheravg_target', 'Fairness_target', 'average_pred', 'otheravg_pred', 'Fairness_pred']

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

    print('Calculate Model Fairness Start')


    df_scored = df.copy()

    df_count = len(df_scored)
    print(f'Total count: {df_count}')

    final_df_list = []
    
    for i, protected_feature in enumerate(protected_groups):
        print(f'Computing fairness scores for {protected_feature}')
        fairness_df = calculate_model_fairness_classwise(df_scored, target_feature, protected_feature, prediction_col)
        
        fairness_df = fairness_df.sort_values(['Protected_Class'])
        final_df_list.append(fairness_df)
    
    # Combine all fairness dataframes
    final_df = pd.concat(final_df_list, ignore_index=True)
    
    final_bnf_df = final_df[['Protected_Feature', 'Protected_Class', 'count', 'average_target', 'otheravg_target', 'Fairness_target', 'average_pred', 'otheravg_pred', 'Fairness_pred']]
    print('Fairness computations for overall population is complete')

    return final_bnf_df



def calculate_model_fairness_classwise(df: pd.DataFrame, target_feature: str, protected_group: str, prediction_col: str) -> pd.DataFrame:
    """
    Calculate model fairness scores for a specific protected group
    Args:
        - df (pd.DataFrame): Input dataframe with model features, target, predicted scores and protected features
        - protected_group (str): Any protected group column name
        - prediction_col (str): column name of predicted scores
        - target_feature (str): column name of targets

    Returns:
        pd.DataFrame: A DataFrame containing fairness scores for a specific protected_group
    """
   
    # Group by protected group and calculate aggregates
    grouped_df = df.groupby(protected_group).agg({
        target_feature: ['count', 'sum'],
        prediction_col: 'sum'
    }).reset_index()
    
    # Flatten column names
    grouped_df.columns = [protected_group, 'count', 'target', 'predicted']
    
    # Calculate totals
    sum_counts = grouped_df['count'].sum()
    sum_preds = grouped_df['predicted'].sum()
    sum_targets = grouped_df['target'].sum()
    
    # Calculate fairness metrics
    grouped_df['average_target'] = grouped_df['target'] / grouped_df['count']
    grouped_df['average_pred'] = grouped_df['predicted'] / grouped_df['count']
    
    grouped_df['otheravg_target'] = (sum_targets - grouped_df['target']) / (sum_counts - grouped_df['count'])
    grouped_df['otheravg_pred'] = (sum_preds - grouped_df['predicted']) / (sum_counts - grouped_df['count'])
    
    # Calculate fairness ratios (taking the smaller ratio to ensure values <= 1)
    def calculate_fairness_ratio(avg_col, other_avg_col):
        ratio1 = avg_col / other_avg_col
        ratio2 = other_avg_col / avg_col
        return np.where(np.abs(ratio1) < np.abs(ratio2), ratio1, ratio2)
    
    # Calculate prediction fairness using ratio-based approach
    grouped_df['Fairness_pred'] = calculate_fairness_ratio(grouped_df['average_pred'], grouped_df['otheravg_pred'])
    
    grouped_df['Fairness_target'] = calculate_fairness_ratio(grouped_df['average_target'], grouped_df['otheravg_target'])
    
    # Rename and select columns
    final_df = grouped_df.rename(columns={protected_group: 'Protected_Class'})
    final_df['Protected_Feature'] = protected_group
    
    # Select final columns
    final_df = final_df[['Protected_Class', 'count', 'Fairness_target', 'average_target', 'otheravg_target', 
                        'Fairness_pred', 'average_pred', 'otheravg_pred', 'Protected_Feature']]
    
    return final_df


def calculate_classification_target_fairness(df: pd.DataFrame, target_feature: str, protected_groups: list):
    """
    Calculate target fairness for classification using chi-square test
    Compares class distributions across protected groups
    
    Args:
        df (pd.DataFrame): Input dataframe with target and protected features
        target_feature (str): column name of target feature (class labels)
        protected_groups (list): List of protected group column names
    
    Returns:
        pd.DataFrame: Fairness results with chi-square based metrics
    """
    
    final_df_list = []
    
    for protected_feature in protected_groups:
        print(f'Computing classification target fairness for {protected_feature}')
        
        # Create contingency table: protected groups vs classes
        contingency_table = pd.crosstab(df[protected_feature], df[target_feature])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate fairness metric for each protected group
        fairness_results = []
        
        for group_name in contingency_table.index:
            # Observed distribution for this group
            observed = contingency_table.loc[group_name].values
            total_group = observed.sum()
            observed_proportions = observed / total_group
            
            # Expected distribution (overall population distribution)
            overall_proportions = df[target_feature].value_counts(normalize=True).sort_index().values
            
            # Calculate chi-square contribution for this group
            expected_counts = overall_proportions * total_group
            chi2_contribution = np.sum((observed - expected_counts)**2 / expected_counts)
            
            # Convert chi-square to fairness score (higher = more fair)
            # Use a more aggressive transformation to achieve target sensitivity
            # For high bias like [0.2, 0.5, 0.3] vs [0.34, 0.33, 0.33], we want score < 0.8
            
            # Normalize chi-square by sample size and degrees of freedom
            chi2_per_sample = chi2_contribution / total_group
            
            # Apply sigmoid-like transformation for better sensitivity
            # Scale factor chosen to map typical bias patterns to desired range
            scale_factor = 10  # Adjust this to control sensitivity
            fairness_score = 1 / (1 + scale_factor * chi2_per_sample)
            
            # Store additional metrics for debugging
            chi2_normalized = chi2_contribution / total_group
            
            fairness_results.append({
                'Protected_Feature': protected_feature,
                'Protected_Class': group_name,
                'count': total_group,
                'observed_proportions': str(observed_proportions.round(3)),
                'expected_proportions': str(overall_proportions.round(3)),
                'chi2_contribution': chi2_contribution,
                'chi2_per_sample': chi2_per_sample,
                'Fairness_target': fairness_score,
                'chi2_pvalue': p_value
            })
        
        group_df = pd.DataFrame(fairness_results)
        final_df_list.append(group_df)
    
    # Combine all results
    final_df = pd.concat(final_df_list, ignore_index=True)
    
    return final_df

    
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
    
    bnf_q = []
    
    if task_type == "regression":
        # Original quantile-based bucketing for regression
        if len(prediction_cols) != 1:
            raise ValueError("For regression, prediction_cols must contain exactly one column")
        
        df_copy['quantile'] = pd.qcut(df_copy[target_feature], q=10, labels=False) + 1
        
        for i in range(1, 11):
            quantile_df = df_copy[df_copy['quantile'] == i].copy()
            q = calculate_model_fairness(quantile_df, target_feature, prediction_col=prediction_cols[0], protected_groups=protected_groups)
            q['bucket'] = f"q{i}"
            q['bucket_type'] = 'quantile'
            bnf_q.append(q)

        bnf = pd.concat(bnf_q, ignore_index=True)
        bnf = bnf.groupby(["Protected_Feature", "Protected_Class"]).apply(weighted_avg).reset_index()
        return bnf
            
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
        pred_bucket_list = []
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


        pred_buckets_df = pd.concat(pred_bucket_list, ignore_index=True)
        bnf_pred = pred_buckets_df.groupby(["Protected_Feature", "Protected_Class"]).apply(weighted_avg).reset_index().drop(columns=["weighted_Fairness_target"])

        # Merge so that final weighted_Fairness_target comes from target view
        # and weighted_Fairness_pred comes from prediction view only
        bnf = bnf_target[["Protected_Feature", "Protected_Class", "weighted_Fairness_target"]].merge(
            bnf_pred,
            on=["Protected_Feature", "Protected_Class"],
            how="inner"
        )

        return bnf
    
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression' or 'classification'")

    # Combine all bucket results (regression only path reaches here)
 