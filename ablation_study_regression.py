"""
Ablation Study for Regression Fairness (3-category, single-class bias)

This script runs three ablations:
  1) Varying amount of injected bias
  2) Varying class distribution (group proportions)
  3) Varying model performance (signal-to-noise)

For each synthetic dataset, it computes fairness using fair_ai_results (regression path).

Outputs:
  - CSV summaries saved under ./ablation_outputs/

Assumptions:
  - fairai_utils.py is available in the Python path (same repo) and exports fair_ai_results.
  - We keep the protected-feature schema consistent with the paper sims:
      UTUP  (Unbiased Target, Unbiased Prediction)
      UTBP  (Unbiased Target, Biased Prediction)  -> pred-only bias (we bias utbp_1)
      BTUP  (Biased Target,  Unbiased Prediction) -> target shift only  (we bias btup_2)
      BTBP  (Biased Target,  Biased Prediction)  -> both (we bias btbp_3)

Directional ratios convention:
  Fairness_target = avg(target)_group / avg(target)_others
  Fairness_pred   = avg(pred)_group   / avg(pred)_others

Usage:
  Run directly: python ablation_regression.py
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Local import (assumes this file sits next to fairai_utils.py in the repo)
from fairai_utils import fair_ai_results

# ----------------------------
# Reproducibility utilities
# ----------------------------
DEFAULT_SEED = 0

def make_rng(seed: int = DEFAULT_SEED):
    return np.random.default_rng(seed)

# ----------------------------
# Synthetic data generator
# ----------------------------
@dataclass
class GenConfig:
    num_rows: int = 6000
    proportions: Tuple[float, float, float] = (1/3, 1/3, 1/3)
    # Bias magnitudes (directional). Values >1 imply higher means for the biased class.
    pred_bias_factor: float = 1.3   # applies to UTBP biased class (prediction only)
    target_bias_factor: float = 1.3 # applies to BTUP biased class (target (and thus pred) shift)
    both_target_factor: float = 0.75 # applies to BTBP biased class (target)
    both_pred_factor: float = 0.6   # applies to BTBP biased class (prediction)
    # Model quality knobs
    pred_alpha: float = 1.0  # how strongly prediction tracks target mean (1.0 ~ follows exactly before noise)
    noise_scale: float = 5.0 # prediction noise std
    base_mu: float = 50.0
    base_sigma: float = 10.0
    seed: int = DEFAULT_SEED


def generate_3cat_single_bias(cfg: GenConfig) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, float, float]]]:
    """Generate a 3-category synthetic regression dataset with single-class biases.

    Returns
    -------
    df : DataFrame with columns [UTUP, UTBP, BTUP, BTBP, target, prediction]
    biased_info : dict mapping protected feature -> (biased_class_name, target_factor, pred_factor)
    """
    rng = make_rng(cfg.seed)

    num_rows = cfg.num_rows
    # protected features (3 categories each)
    groups_utup = [f"utup_{i+1}" for i in range(3)]
    groups_utbp = [f"utbp_{i+1}" for i in range(3)]
    groups_btup = [f"btup_{i+1}" for i in range(3)]
    groups_btbp = [f"btbp_{i+1}" for i in range(3)]

    df = pd.DataFrame({
        'UTUP': rng.choice(groups_utup, num_rows, p=cfg.proportions),
        'UTBP': rng.choice(groups_utbp, num_rows, p=cfg.proportions),
        'BTUP': rng.choice(groups_btup, num_rows, p=cfg.proportions),
        'BTBP': rng.choice(groups_btbp, num_rows, p=cfg.proportions),
    })

    # base target & prediction
    target = rng.normal(cfg.base_mu, cfg.base_sigma, num_rows)
    # Start with prediction that (optionally) shrinks towards 0 + noise
    # We'll later re-apply structure to keep the average relationships clear.
    prediction = cfg.pred_alpha * target + rng.normal(loc=0, scale=cfg.noise_scale, size=num_rows)

    df['target'] = target
    df['prediction'] = prediction

    # -------------------
    # Inject biases
    # -------------------
    biased_info: Dict[str, Tuple[str, float, float]] = {}

    # 1) UTUP: intentionally *no bias*

    # 2) UTBP: pred-only bias for one class (keep target neutral)
    biased_utbp_class = 'utbp_1'
    mask = df['UTBP'] == biased_utbp_class
    df.loc[mask, 'prediction'] *= cfg.pred_bias_factor
    biased_info['UTBP'] = (biased_utbp_class, 1.0, cfg.pred_bias_factor)

    # 3) BTUP: target shift for one class; prediction follows target structure (no *extra* bias)
    biased_btup_class = 'btup_1'
    mask = df['BTUP'] == biased_btup_class
    df.loc[mask, 'target'] *= cfg.target_bias_factor
    df.loc[mask, 'prediction'] *= cfg.target_bias_factor  # model follows target shift
    biased_info['BTUP'] = (biased_btup_class, cfg.target_bias_factor, cfg.target_bias_factor)

    # 4) BTBP: both target and prediction biased with possibly different magnitudes
    biased_btbp_class = 'btbp_1'
    mask = df['BTBP'] == biased_btbp_class
    df.loc[mask, 'target'] *= cfg.both_target_factor
    df.loc[mask, 'prediction'] *= cfg.both_pred_factor
    biased_info['BTBP'] = (biased_btbp_class, cfg.both_target_factor, cfg.both_pred_factor)

    return df, biased_info

# ----------------------------
# Ablation runners
# ----------------------------

def run_fairness(df: pd.DataFrame, protected: List[str]) -> pd.DataFrame:
    """Run fair_ai on regression and return per-group weighted fairness."""
    df = fair_ai_results(
        df.copy(),
        protected_groups=protected,
        target_feature='target',
        prediction_cols=['prediction'],
        task_type='regression',
    )[['Protected_Feature', 'Protected_Class', 'count', 'weighted_Fairness_target', 'weighted_Fairness_pred']]
    return df.rename(columns={"weighted_Fairness_target": "FairAI_target", "weighted_Fairness_pred": "FairAI_pred"})


def run_multiple_seeds(cfg: GenConfig, protected: List[str], num_runs: int = 30) -> pd.DataFrame:
    """Run the same configuration multiple times with different seeds and aggregate results."""
    all_results = []
    
    for run_idx in range(num_runs):
        # Use different seed for each run
        run_cfg = GenConfig(**{**cfg.__dict__, 'seed': cfg.seed + run_idx})
        df, _ = generate_3cat_single_bias(run_cfg)
        result = run_fairness(df, protected)
        result['run'] = run_idx
        all_results.append(result)
    
    # Combine all runs
    combined = pd.concat(all_results, ignore_index=True)
    
    # Calculate statistics for each protected feature and class combination
    stats = combined.groupby(['Protected_Feature', 'Protected_Class']).agg({
        'count': ['mean', 'std'],
        'FairAI_target': ['mean', 'std'],
        'FairAI_pred': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['Protected_Feature', 'Protected_Class', 
                     'count_mean', 'count_std',
                     'FairAI_target_mean', 'FairAI_target_std',
                     'FairAI_pred_mean', 'FairAI_pred_std']
    
    # Calculate standard errors (std / sqrt(n))
    stats['count_se'] = stats['count_std'] / np.sqrt(num_runs)
    stats['FairAI_target_se'] = stats['FairAI_target_std'] / np.sqrt(num_runs)
    stats['FairAI_pred_se'] = stats['FairAI_pred_std'] / np.sqrt(num_runs)
    
    return stats[['Protected_Feature', 'Protected_Class', 'count_mean', 'FairAI_target_mean', 'FairAI_target_se', 'FairAI_pred_mean', 'FairAI_pred_se']]


def ablate_vary_bias(levels: List[Tuple[float,float]], base_cfg: GenConfig, num_runs: int = 30) -> pd.DataFrame:
    rows = []
    for t, p in levels:
        # vary pred-only bias and target-only bias together for a sweep; keep BTBP fixed to show contrast
        cfg = GenConfig(**{**base_cfg.__dict__, 'pred_bias_factor': p, 'target_bias_factor': t, 'both_target_factor': t, 'both_pred_factor': p})
        stats = run_multiple_seeds(cfg, ['UTUP','UTBP','BTUP','BTBP'], num_runs)
        print(f"Finished running fairness for t={t},p={p}")
        stats['ablation'] = 'vary_bias'
        stats['level'] = f"t={t},p={p}"
        rows.append(stats)
    return pd.concat(rows, ignore_index=True)


def ablate_vary_distribution(proportion_sets: List[Tuple[float,float,float]], base_cfg: GenConfig, num_runs: int = 30) -> pd.DataFrame:
    rows = []
    for p in proportion_sets:
        cfg = GenConfig(**{**base_cfg.__dict__, 'proportions': p})
        stats = run_multiple_seeds(cfg, ['UTUP','UTBP','BTUP','BTBP'], num_runs)
        print(f"Finished running fairness for distribution={p}")
        stats['ablation'] = 'vary_distribution'
        stats['level'] = str(tuple(round(pi,3) for pi in p))
        rows.append(stats)
    return pd.concat(rows, ignore_index=True)


def ablate_vary_model(perf_settings: List[Tuple[float,float]], base_cfg: GenConfig, num_runs: int = 30) -> pd.DataFrame:
    """perf_settings: list of (pred_alpha, noise_scale) pairs."""
    rows = []
    for alpha, noise in perf_settings:
        cfg = GenConfig(**{**base_cfg.__dict__, 'pred_alpha': alpha, 'noise_scale': noise})
        stats = run_multiple_seeds(cfg, ['UTUP','UTBP','BTUP','BTBP'], num_runs)
        print(f"Finished running fairness for model={alpha},{noise}")
        stats['ablation'] = 'vary_model'
        stats['level'] = f"a={alpha},b={noise}"
        rows.append(stats)
    return pd.concat(rows, ignore_index=True)

# ----------------------------
# Main
# ----------------------------

def main():
    num_runs = 30  # Number of runs per configuration for statistical analysis
    
    base_cfg = GenConfig(
        num_rows=6000,
        proportions=(1/3, 1/3, 1/3),
        pred_bias_factor=1.3,
        target_bias_factor=1.3,
        both_target_factor=0.75,
        both_pred_factor=0.6,
        pred_alpha=1.0,
        noise_scale=5.0,
        seed=DEFAULT_SEED,
    )

    print(f"Starting ablation study with {num_runs} runs per configuration...")

    # 1) Varying amount of bias (magnitude sweep)
    print("\n1. Running bias variation ablation...")
    bias_levels = [
                (1.0, 1.0), 
                (0.75, 0.75),
                (0.75, 0.6),
                (0.75, 0.5),
                (0.75, 0.4),
                (1.2, 1.4), 
                (1.2, 1.5), 
                (1.2, 1.6), 
                (1.2, 1.7), 
                ]
    out_bias = ablate_vary_bias(bias_levels, base_cfg, num_runs)
    print(f"   Completed bias variation: {len(bias_levels)} configurations × {num_runs} runs")

    # 2) Varying class distribution (group proportions)
    print("\n2. Running distribution variation ablation...")
    proportion_sets = [(1/3,1/3,1/3), (0.5,0.25,0.25), (0.25, 0.25, 0.5), (0.7,0.2,0.1), (0.1, 0.2, 0.7), (0.2, 0.1, 0.7), (0.8,0.19,0.01), (0.01, 0.19, 0.8), (0.19, 0.01, 0.8)]
    out_dist = ablate_vary_distribution(proportion_sets, base_cfg, num_runs)
    print(f"   Completed distribution variation: {len(proportion_sets)} configurations × {num_runs} runs")

    # 3) Varying model performance (signal-to-noise via alpha and noise)
    print("\n3. Running model performance variation ablation...")
    perf_settings = [(1.0,5.0), (0.8,5.0), (1.2,5.0), (1.0,10.0),(0.8,10.0), (1.2,10.0), (1.0,20.0),(0.8,20.0), (1.2,20.0)]
    out_perf = ablate_vary_model(perf_settings, base_cfg, num_runs)
    print(f"   Completed model performance variation: {len(perf_settings)} configurations × {num_runs} runs")

    # Combine results and save
    print("\n4. Combining results and saving...")
    combined = pd.concat([out_bias, out_dist, out_perf], ignore_index=True)
    combined.to_excel('results/reg_ablation.xlsx', index=False)

if __name__ == '__main__':
    main()
