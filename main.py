# Import all functions from utility modules
from fairai_utils import *
from dalex_utils import *
from synthetic_utils import *
import xlsxwriter


def choose_simulation(simulation_type):
    """Choose and run the appropriate simulation function"""
    if simulation_type == "low_bias_single_cat_equal_distribution":
        return low_bias_single_cat_equal_distribution()
    elif simulation_type == "high_bias_single_cat_equal_distribution":
        return high_bias_single_cat_equal_distribution()
    elif simulation_type == "high_bias_single_cat_unequal_distribution":
        return high_bias_single_cat_unequal_distribution()
    elif simulation_type == "high_bias_multiple_cats_unequal_distribution":
        return high_bias_multiple_cats_unequal_distribution()
    elif simulation_type == "high_bias_multiple_cats_unequal_distribution_poor_model":
        return high_bias_multiple_cats_unequal_distribution_poor_model()
    else:
        raise ValueError(f"Unknown simulation type: {simulation_type}")

def get_protected_groups(simulation_type):
    """Get the appropriate protected groups for each simulation type"""
    if "multiple_cats" in simulation_type:
        # Multi-category simulations use extended protected groups
        return ['UTUP', 'UTBP_ML_O', 'BTUP_ML_O', 'BTBP_ML_O', 'UTBP_MH_O', 'BTUP_MH_O', 'BTBP_MH_O', 
                'UTBP_ML_S', 'BTUP_ML_S', 'BTBP_ML_S', 'UTBP_MH_S', 'BTUP_MH_S', 'BTBP_MH_S', 
                'UTBP_LH_O', 'BTUP_LH_O', 'BTBP_LH_O', 'UTBP_LH_S', 'BTUP_LH_S', 'BTBP_LH_S', 
                'UTBP_MM_O', 'BTUP_MM_O', 'BTBP_MM_O', 'UTBP_MM_S', 'BTUP_MM_S', 'BTBP_MM_S']
    else:
        # Single category simulations use simple protected groups
        return ["UTUP", "UTBP", "BTUP", "BTBP"]

if __name__ == "__main__":
    simulation_types = ["low_bias_single_cat_equal_distribution", "high_bias_single_cat_equal_distribution", "high_bias_single_cat_unequal_distribution", "high_bias_multiple_cats_unequal_distribution", "high_bias_multiple_cats_unequal_distribution_poor_model"]
    df_all = []
    for simulation_type in simulation_types:
        print(f"Running {simulation_type}")
        df_dalex = []
        df, biased_categories = choose_simulation(simulation_type)
        
        # Get the correct protected groups for this simulation type
        groups = get_protected_groups(simulation_type)

        # Collect FairAI results for each group
        df_fairai = fair_ai_results(df, groups)
        
        # Collect DALEX results for each group
        for group in groups:
            dalex_group_results = dalex_results(df, group)
            dalex_group_results['Protected_Feature'] = group  # Add protected feature column to match FairAI format
            df_dalex.append(dalex_group_results)
        
        df_dalex = pd.concat(df_dalex, ignore_index=True)
        
        # Join the dataframes based on Protected_Feature and Protected_Class/subgroup
        df_combined = df_fairai.merge(
            df_dalex, 
            left_on=['Protected_Feature', 'Protected_Class'], 
            right_on=['Protected_Feature', 'subgroup'], 
            how='inner'
        )
        
        # Drop the duplicate subgroup column
        df_combined = df_combined.drop(['subgroup', 'weighted_average_target', 'weighted_average_pred', 'weighted_otheravg_target', 'weighted_otheravg_pred'], axis=1)
        
        # Create dictionaries for bias values from biased_categories
        # biased_categories is now a list of (cat_name, target_bias, pred_bias) tuples
        target_bias_dict = {}
        pred_bias_dict = {}
        
        for cat_name, target_bias, pred_bias in biased_categories:
            target_bias_dict[cat_name] = target_bias
            pred_bias_dict[cat_name] = pred_bias
        
        # Add biased column based on whether Protected_Class is in biased category names
        
        # Add target_bias and pred_bias columns with default value 1 for missing categories
        df_combined['target_bias'] = df_combined['Protected_Class'].map(target_bias_dict).fillna(1)
        df_combined['pred_bias'] = df_combined['Protected_Class'].map(pred_bias_dict).fillna(1)
        df_combined['simulation_type'] = simulation_type
        df_combined = df_combined[["simulation_type", "Protected_Feature", "Protected_Class", "target_bias", "pred_bias", "count", "weighted_Fairness_target", "weighted_Fairness_pred", "independence", "separation", "sufficiency"]]
        df_combined = df_combined.rename(columns={"weighted_Fairness_target": "FairAI_target", "weighted_Fairness_pred": "FairAI_pred"})
        df_all.append(df_combined)

    df_combined = pd.concat(df_all, ignore_index=True)
    df_combined.to_excel('results/regression_simulations.xlsx', index=False)

