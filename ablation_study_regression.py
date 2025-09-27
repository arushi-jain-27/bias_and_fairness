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


def ablate_vary_bias(levels: List[Tuple[float,float]], base_cfg: GenConfig) -> pd.DataFrame:
    rows = []
    for t, p in levels:
        # vary pred-only bias and target-only bias together for a sweep; keep BTBP fixed to show contrast
        cfg = GenConfig(**{**base_cfg.__dict__, 'pred_bias_factor': p, 'target_bias_factor': t, 'both_target_factor': t, 'both_pred_factor': p})
        df, info = generate_3cat_single_bias(cfg)
        bnf = run_fairness(df, ['UTUP','UTBP','BTUP','BTBP'])
        bnf['ablation'] = 'vary_bias'
        bnf['level'] = f"target={t},p={p}"
        rows.append(bnf)
    return pd.concat(rows, ignore_index=True)


def ablate_vary_distribution(proportion_sets: List[Tuple[float,float,float]], base_cfg: GenConfig) -> pd.DataFrame:
    rows = []
    for p in proportion_sets:
        cfg = GenConfig(**{**base_cfg.__dict__, 'proportions': p})
        df, info = generate_3cat_single_bias(cfg)
        bnf = run_fairness(df, ['UTUP','UTBP','BTUP','BTBP'])
        bnf['ablation'] = 'vary_distribution'
        bnf['level'] = str(tuple(round(pi,3) for pi in p))
        rows.append(bnf)
    return pd.concat(rows, ignore_index=True)


def ablate_vary_model(perf_settings: List[Tuple[float,float]], base_cfg: GenConfig) -> pd.DataFrame:
    """perf_settings: list of (pred_alpha, noise_scale) pairs."""
    rows = []
    for alpha, noise in perf_settings:
        cfg = GenConfig(**{**base_cfg.__dict__, 'pred_alpha': alpha, 'noise_scale': noise})
        df, info = generate_3cat_single_bias(cfg)
        bnf = run_fairness(df, ['UTUP','UTBP','BTUP','BTBP'])
        bnf['ablation'] = 'vary_model'
        bnf['level'] = f"alpha={alpha},noise={noise}"
        rows.append(bnf)
    return pd.concat(rows, ignore_index=True)

# ----------------------------
# Main
# ----------------------------

def main():

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



    # 1) Varying amount of bias (magnitude sweep)
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
    out_bias = ablate_vary_bias(bias_levels, base_cfg)

    # 2) Varying class distribution (group proportions)
    proportion_sets = [(1/3,1/3,1/3), (0.5,0.25,0.25), (0.25, 0.25, 0.5), (0.7,0.2,0.1), (0.1, 0.2, 0.7), (0.2, 0.1, 0.7), (0.8,0.19,0.01), (0.01, 0.19, 0.8), (0.19, 0.01, 0.8)]
    out_dist = ablate_vary_distribution(proportion_sets, base_cfg)

    # 3) Varying model performance (signal-to-noise via alpha and noise)
    perf_settings = [(1.0,5.0), (0.8,5.0), (1.2,5.0), (1.0,10.0),(0.8,10.0), (1.2,10.0), (1.0,20.0),(0.8,20.0), (1.2,20.0)]
    out_perf = ablate_vary_model(perf_settings, base_cfg)

    # Optional: combined CSV for quick analysis
    combined = pd.concat([out_bias, out_dist, out_perf], ignore_index=True)
    combined.to_excel('results/reg_ablation.xlsx', index=False)

if __name__ == '__main__':
    main()
