# Import all functions from utility modules
from fairai_utils import *
from synthetic_class_utils import *
import pandas as pd



def get_protected_groups_classification(simulation_type):
    """Get the appropriate protected groups for each classification simulation type"""
    # For now, all classification simulations use simple protected groups
    return ["UTUP", "UTBP", "BTUP", "BTBP"]

if __name__ == "__main__":
    simulation_types = [
        "high_bias_single_cat_equal_distribution_classification", 
        "high_bias_single_cat_unequal_distribution_classification", 
        "high_bias_single_cat_equal_distribution_classification_poor_model"
    ]
    
    df_all = []
    
    for simulation_type in simulation_types:
        print(f"Running classification simulation: {simulation_type}")
        df, biased_categories = choose_classification_simulation(simulation_type)
        
        # Get the correct protected groups for this simulation type
        groups = get_protected_groups_classification(simulation_type)

        # Collect FairAI results for classification
        df_fairai = fair_ai_results(
            df, 
            groups, 
            prediction_cols=["prob_class_0", "prob_class_1", "prob_class_2"], 
            task_type="classification"
        )
        
        # Create dictionaries for bias values from biased_categories
        # biased_categories is now a list of (cat_name, target_bias, pred_bias) tuples
        # For classification, these are probability distributions instead of scalars
        target_bias_dict = {}
        pred_bias_dict = {}
        
        for cat_name, target_bias, pred_bias in biased_categories:
            # Convert tuples to strings for Excel compatibility
            target_bias_dict[cat_name] = str(target_bias)
            pred_bias_dict[cat_name] = str(pred_bias)
        
        # Add target_bias and pred_bias columns with default value for missing categories
        df_fairai['target_bias'] = df_fairai['Protected_Class'].map(target_bias_dict).fillna(str((0.34, 0.33, 0.33)))  # Default to baseline
        df_fairai['pred_bias'] = df_fairai['Protected_Class'].map(pred_bias_dict).fillna(str((1, 1, 1)))  # Default to no bias
        df_fairai['simulation_type'] = simulation_type
        
        # Select and rename columns
        df_combined = df_fairai[["simulation_type", "Protected_Feature", "Protected_Class", "target_bias", "pred_bias", "count", "weighted_Fairness_target", "weighted_Fairness_pred"]]
        df_combined = df_combined.rename(columns={"weighted_Fairness_target": "FairAI_target", "weighted_Fairness_pred": "FairAI_pred"})
        df_all.append(df_combined)

    # Combine all results
    df_combined = pd.concat(df_all, ignore_index=True)
    df_combined.to_excel('results/classification_simulations.xlsx', index=False)
