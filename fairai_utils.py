import pandas as pd
import numpy as np


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
    
    grouped_df['Fairness_target'] = calculate_fairness_ratio(grouped_df['average_target'], grouped_df['otheravg_target'])
    grouped_df['Fairness_pred'] = calculate_fairness_ratio(grouped_df['average_pred'], grouped_df['otheravg_pred'])
    
    # Rename and select columns
    final_df = grouped_df.rename(columns={protected_group: 'Protected_Class'})
    final_df['Protected_Feature'] = protected_group
    
    # Select final columns
    final_df = final_df[['Protected_Class', 'count', 'Fairness_target', 'average_target', 'otheravg_target', 
                        'Fairness_pred', 'average_pred', 'otheravg_pred', 'Protected_Feature']]
    
    return final_df

    
def fair_ai_results(df, protected_groups, target_feature = "target", prediction_col="prediction"):
    """
    Calculate fairness results across quantiles of the target feature
    Args:
        - df (pd.DataFrame): Input dataframe
        - target_feature (str): column name of target feature
        - protected_groups (list): List of protected group column names
        - prediction_col (str): column name of predictions (default: "prediction")
    
    Returns:
        pd.DataFrame: Weighted fairness results across quantiles
    """
    
    df_copy = df.copy()
    df_copy['quantile'] = pd.qcut(df_copy[target_feature], q=10, labels=False) + 1
    
    bnf_q = []
    for i in range(1, 11):
        quantile_df = df_copy[df_copy['quantile'] == i].copy()
        q = calculate_model_fairness(quantile_df, target_feature, prediction_col=prediction_col, protected_groups=protected_groups)
        q['quantile'] = f"q{i}"
        bnf_q.append(q)

    # Combine all quantile results
    bnf = pd.concat(bnf_q, ignore_index=True)
    
    # Apply weighted averaging
    bnf = bnf.groupby(["Protected_Feature", "Protected_Class"]).apply(weighted_avg).reset_index()
    
    return bnf