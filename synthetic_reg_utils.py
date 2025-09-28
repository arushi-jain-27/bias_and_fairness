import numpy as np
import pandas as pd


def choose_simulation(simulation_type, seed=0):
    if simulation_type == "low_bias_single_cat_equal_distribution":
        return generate_single_cat_regression_simulation(
            seed=seed,
            proportions=[1/3, 1/3, 1/3],  # Equal distribution
            utbp_bias=1.2,  # Low bias
            btup_bias=1.2,
            btbp_target_bias=0.85,
            btbp_pred_bias=0.75
        )
    elif simulation_type == "high_bias_single_cat_equal_distribution":
        return generate_single_cat_regression_simulation(
            seed=seed,
            proportions=[1/3, 1/3, 1/3],  # Equal distribution
            utbp_bias=1.3,  # High bias
            btup_bias=1.3,
            btbp_target_bias=0.75,
            btbp_pred_bias=0.6
        )
    elif simulation_type == "high_bias_single_cat_unequal_distribution":
        return generate_single_cat_regression_simulation(
            seed=seed,
            proportions=[0.4, 0.4, 0.2],  # Unequal distribution
            utbp_bias=1.3,  # High bias
            btup_bias=1.3,
            btbp_target_bias=0.75,
            btbp_pred_bias=0.6
        )
    elif simulation_type == "high_bias_multiple_cats_unequal_distribution":
        return generate_multi_cat_regression_simulation(
            seed=seed,
            prediction_scale=5  # Good model
        )
    elif simulation_type == "high_bias_multiple_cats_unequal_distribution_poor_model":
        return generate_multi_cat_regression_simulation(
            seed=seed,
            prediction_scale=20  # Poor model
        )

def generate_single_cat_regression_simulation(seed=0, proportions=[1/3, 1/3, 1/3], utbp_bias=1.2, btup_bias=1.2, btbp_target_bias=0.85, btbp_pred_bias=0.75):
    """
    Unified single category regression simulation function with configurable parameters.
    
    Parameters:
    - seed: Random seed for reproducibility
    - proportions: Group proportions for distribution
    - utbp_bias: Bias multiplier for UTBP prediction
    - btup_bias: Bias multiplier for BTUP target and prediction
    - btbp_target_bias: Bias multiplier for BTBP target
    - btbp_pred_bias: Bias multiplier for BTBP prediction
    """
    num_rows = 6000
    num_groups = 3
    protected_groups = ['UTUP', 'UTBP', 'BTUP', 'BTBP']
    groups_utup = [f"utup_{i+1}" for i in range(num_groups)]
    groups_utbp = [f"utbp_{i+1}" for i in range(num_groups)]
    groups_btup = [f"btup_{i+1}" for i in range(num_groups)]
    groups_btbp = [f"btbp_{i+1}" for i in range(num_groups)]

    rng = np.random.default_rng(seed)

    data = {
        'UTUP': rng.choice(groups_utup, num_rows, p=proportions),
        'UTBP': rng.choice(groups_utbp, num_rows, p=proportions),
        'BTUP': rng.choice(groups_btup, num_rows, p=proportions),
        'BTBP': rng.choice(groups_btbp, num_rows, p=proportions),
    }
    df = pd.DataFrame(data)

    # Generate target column - normally distributed
    df['target'] = rng.normal(loc=50, scale=10, size=num_rows)
    # Generate prediction column - initially close to target
    df['prediction'] = df['target'] + rng.normal(loc=0, scale=5, size=num_rows)

    # Apply bias conditions:

    # 1. Unbiased for UTUP
    # No changes needed as both target and prediction should be unbiased.

    # 2. Target is unbiased, but prediction is biased for 1 category in UTBP
    biased_utbp_category = 'utbp_1'
    df.loc[df['UTBP'] == biased_utbp_category, 'prediction'] *= utbp_bias

    # 3. Target is biased, but prediction is unbiased for 1 category in BTUP
    biased_btup_category = 'btup_1'
    df.loc[df['BTUP'] == biased_btup_category, ['target', 'prediction']] *= btup_bias

    # 4. Both target and prediction are biased for BTBP
    biased_btbp_category = 'btbp_1'
    df.loc[df['BTBP'] == biased_btbp_category, 'target'] *= btbp_target_bias
    df.loc[df['BTBP'] == biased_btbp_category, 'prediction'] *= btbp_pred_bias

    return df, [(biased_utbp_category, 1, utbp_bias), (biased_btup_category, btup_bias, btup_bias), (biased_btbp_category, btbp_target_bias, btbp_pred_bias)]





def generate_multi_cat_regression_simulation(seed=0, prediction_scale=5):
    """
    Unified multiple category regression simulation function with configurable parameters.
    
    Parameters:
    - seed: Random seed for reproducibility
    - prediction_scale: Standard deviation for prediction noise (5 for good model, 20 for poor model)
    """
    num_rows = 60000
    num_groups = 4
    proportions = [0.25,0.25,0.45, 0.05]
    protected_groups = ['UTUP', 'UTBP_ML_O', 'BTUP_ML_O', 'BTBP_ML_O', 'UTBP_MH_O', 'BTUP_MH_O', 'BTBP_MH_O', 'UTBP_ML_S', 'BTUP_ML_S', 'BTBP_ML_S', 'UTBP_MH_S',  'BTUP_MH_S', 'BTBP_MH_S', 'UTBP_LH_O', 'BTUP_LH_O', 'BTBP_LH_O', 'UTBP_LH_S', 'BTUP_LH_S', 'BTBP_LH_S', 'UTBP_MM_O', 'BTUP_MM_O', 'BTBP_MM_O', 'UTBP_MM_S', 'BTUP_MM_S', 'BTBP_MM_S']
    groups_utup = [f"utup_{i+1}" for i in range(num_groups)]
    groups_utbp_ml_o = [f"utbp_ml_o_{i+1}" for i in range(num_groups)]
    groups_btup_ml_o = [f"btup_ml_o_{i+1}" for i in range(num_groups)]
    groups_btbp_ml_o = [f"btbp_ml_o_{i+1}" for i in range(num_groups)]
    groups_utbp_mh_o = [f"utbp_mh_o_{i+1}" for i in range(num_groups)]
    groups_btup_mh_o = [f"btup_mh_o_{i+1}" for i in range(num_groups)]
    groups_btbp_mh_o = [f"btbp_mh_o_{i+1}" for i in range(num_groups)]
    groups_utbp_ml_s = [f"utbp_ml_s_{i+1}" for i in range(num_groups)]
    groups_btup_ml_s = [f"btup_ml_s_{i+1}" for i in range(num_groups)]
    groups_btbp_ml_s = [f"btbp_ml_s_{i+1}" for i in range(num_groups)]
    groups_utbp_mh_s = [f"utbp_mh_s_{i+1}" for i in range(num_groups)]
    groups_btup_mh_s = [f"btup_mh_s_{i+1}" for i in range(num_groups)]
    groups_btbp_mh_s = [f"btbp_mh_s_{i+1}" for i in range(num_groups)]
    groups_utbp_lh_o = [f"utbp_lh_o_{i+1}" for i in range(num_groups)]
    groups_btup_lh_o = [f"btup_lh_o_{i+1}" for i in range(num_groups)]
    groups_btbp_lh_o = [f"btbp_lh_o_{i+1}" for i in range(num_groups)]
    groups_utbp_lh_s = [f"utbp_lh_s_{i+1}" for i in range(num_groups)]
    groups_btup_lh_s = [f"btup_lh_s_{i+1}" for i in range(num_groups)]
    groups_btbp_lh_s = [f"btbp_lh_s_{i+1}" for i in range(num_groups)]
    groups_utbp_mm_o = [f"utbp_mm_o_{i+1}" for i in range(num_groups)]
    groups_btup_mm_o = [f"btup_mm_o_{i+1}" for i in range(num_groups)]
    groups_btbp_mm_o = [f"btbp_mm_o_{i+1}" for i in range(num_groups)]
    groups_utbp_mm_s = [f"utbp_mm_s_{i+1}" for i in range(num_groups)]
    groups_btup_mm_s = [f"btup_mm_s_{i+1}" for i in range(num_groups)]
    groups_btbp_mm_s = [f"btbp_mm_s_{i+1}" for i in range(num_groups)]

    rng = np.random.default_rng(seed)

    data = {
        'UTUP': rng.choice(groups_utup, num_rows, p=proportions),
        'UTBP_ML_O': rng.choice(groups_utbp_ml_o, num_rows, p=proportions),
        'BTUP_ML_O': rng.choice(groups_btup_ml_o, num_rows, p=proportions),
        'BTBP_ML_O': rng.choice(groups_btbp_ml_o, num_rows, p=proportions),
        'UTBP_ML_S': rng.choice(groups_utbp_ml_s, num_rows, p=proportions),
        'BTUP_ML_S': rng.choice(groups_btup_ml_s, num_rows, p=proportions),
        'BTBP_ML_S': rng.choice(groups_btbp_ml_s, num_rows, p=proportions),
        'UTBP_MH_O': rng.choice(groups_utbp_mh_o, num_rows, p=proportions),
        'BTUP_MH_O': rng.choice(groups_btup_mh_o, num_rows, p=proportions),
        'BTBP_MH_O': rng.choice(groups_btbp_mh_o, num_rows, p=proportions),
        'UTBP_MH_S': rng.choice(groups_utbp_mh_s, num_rows, p=proportions),
        'BTUP_MH_S': rng.choice(groups_btup_mh_s, num_rows, p=proportions),
        'BTBP_MH_S': rng.choice(groups_btbp_mh_s, num_rows, p=proportions),
        'UTBP_LH_O': rng.choice(groups_utbp_lh_o, num_rows, p=proportions),
        'BTUP_LH_O': rng.choice(groups_btup_lh_o, num_rows, p=proportions),
        'BTBP_LH_O': rng.choice(groups_btbp_lh_o, num_rows, p=proportions),
        'UTBP_LH_S': rng.choice(groups_utbp_lh_s, num_rows, p=proportions),
        'BTUP_LH_S': rng.choice(groups_btup_lh_s, num_rows, p=proportions),
        'BTBP_LH_S': rng.choice(groups_btbp_lh_s, num_rows, p=proportions),
        'UTBP_MM_O': rng.choice(groups_utbp_mm_o, num_rows, p=proportions),
        'BTUP_MM_O': rng.choice(groups_btup_mm_o, num_rows, p=proportions),
        'BTBP_MM_O': rng.choice(groups_btbp_mm_o, num_rows, p=proportions),
        'UTBP_MM_S': rng.choice(groups_utbp_mm_s, num_rows, p=proportions),
        'BTUP_MM_S': rng.choice(groups_btup_mm_s, num_rows, p=proportions),
        'BTBP_MM_S': rng.choice(groups_btbp_mm_s, num_rows, p=proportions),
    }
    df = pd.DataFrame(data)

    # Generate target column - normally distributed
    df['target'] = rng.normal(loc=50, scale=10, size=num_rows)
    # Generate prediction column - initially close to target
    df['prediction'] = df['target'] + rng.normal(loc=0, scale=prediction_scale, size=num_rows)

    # Apply bias conditions:
    biased_categories = []

    def add_biases(df, group, c1, c2, h, l, hbar, lbar):
        cat = group.lower()
        # 1. Unbiased for UTUP
        # No changes needed as both target and prediction should be unbiased.

        # 2. Target is unbiased, but prediction is biased
        biased_category_1 = f'utbp_{cat}_{c1}'
        biased_category_2 = f'utbp_{cat}_{c2}'

        df.loc[df[f'UTBP_{group}'] == biased_category_1, 'prediction'] *= h  # Introduce bias in prediction
        df.loc[df[f'UTBP_{group}'] == biased_category_2, 'prediction'] *= l  # Introduce bias in prediction
        biased_categories.append((biased_category_1, 1, h))
        biased_categories.append((biased_category_2, 1, l))
        # 3. Target is biased, but prediction is unbiased
        biased_category_1 = f'btup_{cat}_{c1}'
        biased_category_2 = f'btup_{cat}_{c2}'
        df.loc[df[f'BTUP_{group}'] == biased_category_1, ['target', 'prediction']] *= h  # Introduce bias in target but prediction still follows the target
        df.loc[df[f'BTUP_{group}'] == biased_category_2, ['target', 'prediction']] *= l  # Introduce bias in target but prediction still follows the target
        biased_categories.append((biased_category_1, h, h))
        biased_categories.append((biased_category_2, l, l))

        # 4. Both target and prediction are biased for BTBP
        biased_category_1 = f'btbp_{cat}_{c1}'
        biased_category_2 = f'btbp_{cat}_{c2}'
        df.loc[df[f'BTBP_{group}'] == biased_category_1, 'target'] *= l  # Introduce bias in prediction
        df.loc[df[f'BTBP_{group}'] == biased_category_1, 'prediction'] *= lbar  # Introduce bias in prediction
        df.loc[df[f'BTBP_{group}'] == biased_category_2, 'target'] *= h  # Introduce bias in prediction
        df.loc[df[f'BTBP_{group}'] == biased_category_2, 'prediction'] *= hbar  # Introduce bias in prediction
        biased_categories.append((biased_category_1, l, lbar))
        biased_categories.append((biased_category_2, h, hbar))

        return df

    df = add_biases(df, "ML_O", 1, 4, 1.2, 0.85, 1.45, 0.65)
    df = add_biases(df, "ML_S", 1, 4, 1.2, 1.4, 1.45, 1.1)
    df = add_biases(df, "MH_O", 1, 3, 1.2, 0.85, 1.45, 0.65)
    df = add_biases(df, "MH_S", 1, 3, 1.2, 1.4, 1.45, 1.1)
    df = add_biases(df, "LH_O", 3, 4, 1.2, 0.85, 1.45, 0.65)
    df = add_biases(df, "LH_S", 3, 4, 1.2, 1.4, 1.45, 1.1)
    df = add_biases(df, "MM_O", 1, 2, 1.2, 0.85, 1.45, 0.65)
    df = add_biases(df, "MM_S", 1, 2, 1.2, 1.4, 1.45, 1.1)
    return df, biased_categories





