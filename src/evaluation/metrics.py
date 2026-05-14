"""
Probabilistic Forecast Metrics
================================
The standard metric trio for interval forecasts:

1. WINKLER SCORE — primary optimization target
   Penalizes BOTH wide intervals AND coverage failures.
   Lower is better. Use this, not RMSE.

2. PICP (Prediction Interval Coverage Probability)
   Empirical coverage rate. Should match nominal level.

3. MPIW (Mean Prediction Interval Width)
   Sharpness. Given equal coverage, narrower = better.

4. CRPS (Continuous Ranked Probability Score)
   Proper scoring rule for distributional forecasts.
   Generalizes MAE to probability distributions.

5. Reliability Diagram
   Plots empirical coverage vs nominal coverage across quantiles.
   Well-calibrated model lies on the diagonal.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Winkler Score for (1-alpha) prediction intervals.
    
    W = (upper - lower) + (2/alpha) * penalty_for_misses
    
    Misses below lower: add (2/alpha) * (lower - y)
    Misses above upper: add (2/alpha) * (y - upper)
    
    Why this metric?
    - Pure interval width → ignores coverage failures
    - Pure coverage → ignores sharpness
    - Winkler jointly optimizes both
    
    Lower is better. Perfect calibration + minimum width = 0 (theoretical).
    """
    width = upper - lower
    
    below_mask = y_true < lower
    above_mask = y_true > upper
    
    penalty = np.zeros_like(y_true, dtype=float)
    penalty[below_mask] = (2 / alpha) * (lower[below_mask] - y_true[below_mask])
    penalty[above_mask] = (2 / alpha) * (y_true[above_mask] - upper[above_mask])
    
    scores = width + penalty
    return float(np.mean(scores))


def picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Prediction Interval Coverage Probability.
    Should match the nominal coverage level (e.g., 0.90 for 90% intervals).
    """
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width. Sharpness metric."""
    return float(np.mean(upper - lower))


def coverage_width_criterion(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eta: float = 0.5,
) -> float:
    """
    CWC (Coverage Width Criterion) — penalizes coverage failures 
    exponentially while rewarding narrow intervals.
    
    CWC = MPIW * (1 + e^(eta * max(0, nominal_cov - empirical_cov)))
    """
    emp_cov = picp(y_true, lower, upper)
    width = mpiw(lower, upper)
    nominal_cov = 0.90  # Assumed; pass as param for generality
    
    if emp_cov >= nominal_cov:
        return width
    
    return width * (1 + np.exp(eta * (nominal_cov - emp_cov)))


def crps_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    CRPS for Gaussian predictive distributions.
    CRPS = sigma * (z * (2Φ(z)-1) + 2φ(z) - 1/√π)
    where z = (y - mu) / sigma
    
    CRPS = MAE when sigma → 0 (reduces to point forecast scoring).
    """
    from scipy.stats import norm
    
    z = (y_true - mu) / (sigma + 1e-8)
    
    crps = sigma * (
        z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    
    return float(np.mean(crps))


def reliability_diagram_data(
    y_true: np.ndarray,
    quantile_preds: Dict[float, np.ndarray],
) -> Tuple[List[float], List[float]]:
    """
    Compute (nominal, empirical) coverage pairs for reliability diagram.
    
    A well-calibrated model: empirical ≈ nominal (on diagonal).
    Overconfident model: empirical < nominal (below diagonal).
    Underconfident model: empirical > nominal (above diagonal).
    
    Returns:
        nominal: list of quantile levels [0.05, 0.10, ..., 0.95]
        empirical: list of actual coverage rates
    """
    sorted_qs = sorted(quantile_preds.keys())
    nominal = []
    empirical = []
    
    for q in sorted_qs:
        preds = quantile_preds[q]
        emp = float(np.mean(y_true <= preds))
        nominal.append(q)
        empirical.append(emp)
    
    return nominal, empirical


def horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    horizons: List[int] = [1, 6, 24, 168],
    alpha: float = 0.10,
) -> Dict[str, Dict]:
    """
    Compute all metrics stratified by forecast horizon.
    
    Winkler scores GROW with horizon (wider intervals needed).
    Never compare raw Winkler across horizons — stratify.
    """
    n = len(y_true)
    results = {}
    
    for h in horizons:
        # Sample every h-th observation to get h-step-ahead forecasts
        indices = np.arange(h - 1, n, h)
        if len(indices) < 10:
            continue
        
        y_h = y_true[indices]
        yp_h = y_pred[indices]
        l_h = lower[indices]
        u_h = upper[indices]
        
        results[f"h={h}h"] = {
            "winkler": round(winkler_score(y_h, l_h, u_h, alpha), 2),
            "picp": round(picp(y_h, l_h, u_h), 4),
            "mpiw": round(mpiw(l_h, u_h), 2),
            "rmse": round(float(np.sqrt(np.mean((y_h - yp_h) ** 2))), 2),
            "mae": round(float(np.mean(np.abs(y_h - yp_h))), 2),
            "n": len(indices),
        }
    
    return results


def full_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_80: np.ndarray,
    upper_80: np.ndarray,
    lower_95: np.ndarray,
    upper_95: np.ndarray,
    quantile_preds: Optional[Dict[float, np.ndarray]] = None,
) -> Dict:
    """
    Generate complete evaluation report.
    Used by evaluate.py and the API response.
    """
    report = {
        "overall": {
            "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 2),
            "mae": round(float(np.mean(np.abs(y_true - y_pred))), 2),
            "mape": round(float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100), 2),
        },
        "interval_80": {
            "winkler": round(winkler_score(y_true, lower_80, upper_80, alpha=0.20), 2),
            "picp": round(picp(y_true, lower_80, upper_80), 4),
            "mpiw": round(mpiw(lower_80, upper_80), 2),
            "nominal_coverage": 0.80,
        },
        "interval_95": {
            "winkler": round(winkler_score(y_true, lower_95, upper_95, alpha=0.05), 2),
            "picp": round(picp(y_true, lower_95, upper_95), 4),
            "mpiw": round(mpiw(lower_95, upper_95), 2),
            "nominal_coverage": 0.95,
        },
        "horizon_metrics_80": horizon_metrics(y_true, y_pred, lower_80, upper_80),
    }
    
    if quantile_preds is not None:
        nominal, empirical = reliability_diagram_data(y_true, quantile_preds)
        ece = np.mean(np.abs(np.array(nominal) - np.array(empirical)))
        report["calibration"] = {
            "nominal": nominal,
            "empirical": empirical,
            "ece": round(float(ece), 4),
        }
    
    return report
