import numpy as np
import pandas as pd


def choose_classification_simulation(simulation_type, seed=0):
    """Choose and run the appropriate classification simulation function"""
    if simulation_type == "high_bias_single_cat_equal_distribution_classification":
        return generate_classification_simulation(
            seed=seed, 
            proportions=[1/3, 1/3, 1/3],  # Equal distribution
            correct_alpha=7, correct_beta=2,  # Well-calibrated
            incorrect_alpha=1.5, incorrect_beta=6
        )
    elif simulation_type == "high_bias_single_cat_unequal_distribution_classification":
        return generate_classification_simulation(
            seed=seed, 
            proportions=[0.4, 0.4, 0.2],  # Unequal distribution
            correct_alpha=7, correct_beta=2,  # Well-calibrated
            incorrect_alpha=1.5, incorrect_beta=6
        )
    elif simulation_type == "high_bias_single_cat_equal_distribution_classification_poor_model":
        return generate_classification_simulation(
            seed=seed, 
            proportions=[1/3, 1/3, 1/3],  # Equal distribution
            correct_alpha=4, correct_beta=3,  # Poorly-calibrated
            incorrect_alpha=2, incorrect_beta=4
        )
    else:
        raise ValueError(f"Unknown classification simulation type: {simulation_type}")


def generate_classification_simulation(seed=0, proportions=None, correct_alpha=7, correct_beta=2, incorrect_alpha=1.5, incorrect_beta=6):
    """
    Unified classification simulation function with configurable parameters.
    
    Parameters:
    - seed: Random seed for reproducibility
    - proportions: Group proportions for unequal distribution. If None, uses equal distribution
    - correct_alpha, correct_beta: Beta distribution parameters for correct class predictions
    - incorrect_alpha, incorrect_beta: Beta distribution parameters for incorrect class predictions
    """
    num_rows = 6000
    num_groups = 3
    protected_groups = ['UTUP', 'UTBP', 'BTUP', 'BTBP']
    groups_utup = [f"utup_{i+1}" for i in range(num_groups)]
    groups_utbp = [f"utbp_{i+1}" for i in range(num_groups)]
    groups_btup = [f"btup_{i+1}" for i in range(num_groups)]
    groups_btbp = [f"btbp_{i+1}" for i in range(num_groups)]

    rng = np.random.default_rng(seed)

    # Generate group assignments
    data = {}
    for group_name, group_list in [('UTUP', groups_utup), ('UTBP', groups_utbp), 
                                  ('BTUP', groups_btup), ('BTBP', groups_btbp)]:

        data[group_name] = rng.choice(group_list, num_rows, p=proportions)
    
    df = pd.DataFrame(data)

    # Generate target column - 3 classes with balanced distribution
    df['target'] = rng.choice([0, 1, 2], num_rows, p=[0.34, 0.33, 0.33])
    
    # Generate base probabilities using provided parameters
    for class_idx in [0, 1, 2]:
        df[f'prob_class_{class_idx}'] = np.where(
            df['target'] == class_idx,
            rng.beta(correct_alpha, correct_beta, num_rows),
            rng.beta(incorrect_alpha, incorrect_beta, num_rows)
        )
    
    # Normalize probabilities to sum to 1
    prob_cols = ['prob_class_0', 'prob_class_1', 'prob_class_2']
    df[prob_cols] = df[prob_cols].div(df[prob_cols].sum(axis=1), axis=0)

    # Apply bias conditions:

    # 1. Unbiased for UTUP - No changes needed

    # 2. Target is unbiased, but prediction is biased for 1 category in UTBP
    biased_utbp_category = 'utbp_1'
    utbp_mask = df['UTBP'] == biased_utbp_category
    # Reduce confidence for class 2, redistribute to classes 0 and 1
    df.loc[utbp_mask, 'prob_class_0'] *= 1.8
    df.loc[utbp_mask, 'prob_class_1'] *= 2.0
    df.loc[utbp_mask, 'prob_class_2'] *= 0.3 

    # Renormalize
    df.loc[utbp_mask, prob_cols] = df.loc[utbp_mask, prob_cols].div(
        df.loc[utbp_mask, prob_cols].sum(axis=1), axis=0
    )

    # 3. Target is biased, but prediction follows the bias for 1 category in BTUP
    biased_btup_category = 'btup_1'
    btup_mask = df['BTUP'] == biased_btup_category
    # First bias the target: increase class 1 representation for this group
    btup_indices = df[btup_mask].index
    df.loc[btup_indices, 'target'] = rng.choice([0, 1, 2], len(btup_indices), p=[0.2, 0.5, 0.3])
    
    # Regenerate probabilities to match new biased targets
    for class_idx in [0, 1, 2]:
        df.loc[btup_mask, f'prob_class_{class_idx}'] = np.where(
            df.loc[btup_mask, 'target'] == class_idx,
            rng.beta(correct_alpha, correct_beta, btup_mask.sum()),
            rng.beta(incorrect_alpha, incorrect_beta, btup_mask.sum())
        )
    # Renormalize
    df.loc[btup_mask, prob_cols] = df.loc[btup_mask, prob_cols].div(
        df.loc[btup_mask, prob_cols].sum(axis=1), axis=0
    )

    # 4. Both target and prediction are biased for BTBP
    biased_btbp_category = 'btbp_1'
    btbp_mask = df['BTBP'] == biased_btbp_category
    # First bias target (reduce class 0 representation)
    btbp_indices = df[btbp_mask].index
    df.loc[btbp_indices, 'target'] = rng.choice([0, 1, 2], len(btbp_indices), p=[0.2, 0.5, 0.3])
    
    # Regenerate probabilities to match new biased targets
    for class_idx in [0, 1, 2]:
        df.loc[btbp_mask, f'prob_class_{class_idx}'] = np.where(
            df.loc[btbp_mask, 'target'] == class_idx,
            rng.beta(correct_alpha, correct_beta, btbp_mask.sum()),
            rng.beta(incorrect_alpha, incorrect_beta, btbp_mask.sum())
        )
    # Renormalize after regeneration
    df.loc[btbp_mask, prob_cols] = df.loc[btbp_mask, prob_cols].div(
        df.loc[btbp_mask, prob_cols].sum(axis=1), axis=0
    )
    
    # Then add additional prediction bias (amplify the target bias)
    df.loc[btbp_mask, 'prob_class_0'] *= 0.5   # Further reduce class 0 confidence
    df.loc[btbp_mask, 'prob_class_1'] *= 3   # Increase class 1 confidence  
    df.loc[btbp_mask, 'prob_class_2'] *= 1.5   # Increase class 2 confidence
    # Final renormalization
    df.loc[btbp_mask, prob_cols] = df.loc[btbp_mask, prob_cols].div(
        df.loc[btbp_mask, prob_cols].sum(axis=1), axis=0
    )

    # Return bias information: (category_name, target_bias_indicator, pred_bias_indicator)
    # For classification, bias indicators represent the bias pattern applied
    return df, [(biased_utbp_category, [0.33,0.33,0.33], [1.8, 2.0, 0.3]), (biased_btup_category, [0.2, 0.5, 0.3], [1,1,1]), (biased_btbp_category, [0.2, 0.5, 0.3], [0.5, 3, 1.5])]

