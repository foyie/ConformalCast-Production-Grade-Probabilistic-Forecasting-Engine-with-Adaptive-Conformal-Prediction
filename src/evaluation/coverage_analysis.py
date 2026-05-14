"""
Coverage Breakdown Analysis
============================
Identifies WHERE your model loses coverage (by horizon, hour, magnitude).
Helps diagnose model weaknesses.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.evaluation.metrics import picp, mpiw


def coverage_by_horizon(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    horizons: List[int] = [1, 6, 24, 168],
) -> Dict[str, Dict]:
    """
    Stratify coverage by forecast horizon.
    
    CRITICAL: Winkler scores are NOT comparable across horizons.
    Always stratify when diagnosing.
    """
    n = len(y_true)
    results = {}
    
    for h in horizons:
        indices = list(range(h - 1, n, h))
        if len(indices) < 10:
            continue
        
        y_h = y_true[indices]
        l_h = lower[indices]
        u_h = upper[indices]
        
        cov = picp(y_h, l_h, u_h)
        width = mpiw(l_h, u_h)
        
        # Count misses by direction
        below = np.sum(y_h < l_h)
        above = np.sum(y_h > u_h)
        
        results[f"h={h}h"] = {
            "coverage": float(cov),
            "interval_width": float(width),
            "n_samples": len(indices),
            "misses_below": int(below),
            "misses_above": int(above),
            "miss_rate": float((below + above) / len(indices)),
        }
    
    return results


def coverage_by_hour_of_day(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    datetime_index: Optional[pd.DatetimeIndex] = None,
) -> Dict[int, Dict]:
    """
    Coverage stratified by hour of day (0-23).
    
    Energy load has strong daily pattern. Check if intervals are
    good at night vs day, peak vs base load times.
    """
    if datetime_index is None:
        print("Warning: No datetime index. Using modulo indexing (may be wrong).")
        datetime_index = pd.date_range("2020-01-01", periods=len(y_true), freq="h")
    
    hours = datetime_index.hour.values
    results = {}
    
    for hour in range(24):
        mask = hours == hour
        if np.sum(mask) < 10:
            continue
        
        y_h = y_true[mask]
        l_h = lower[mask]
        u_h = upper[mask]
        
        cov = picp(y_h, l_h, u_h)
        width = mpiw(l_h, u_h)
        below = np.sum(y_h < l_h)
        above = np.sum(y_h > u_h)
        
        results[hour] = {
            "coverage": float(cov),
            "interval_width": float(width),
            "n_samples": int(np.sum(mask)),
            "misses_below": int(below),
            "misses_above": int(above),
        }
    
    return results


def coverage_by_magnitude(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, Dict]:
    """
    Coverage stratified by load magnitude (low, medium, high, etc).
    
    Often: small loads easy to predict, large spikes hard.
    Check if your intervals adapt to magnitude.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y_true, quantiles)
    
    results = {}
    
    for i in range(n_bins):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if np.sum(mask) < 10:
            continue
        
        y_bin = y_true[mask]
        l_bin = lower[mask]
        u_bin = upper[mask]
        
        cov = picp(y_bin, l_bin, u_bin)
        width = mpiw(l_bin, u_bin)
        
        bin_label = f"Bin {i+1}: {bins[i]:.0f}-{bins[i+1]:.0f}"
        results[bin_label] = {
            "coverage": float(cov),
            "interval_width": float(width),
            "n_samples": int(np.sum(mask)),
            "load_range": (float(bins[i]), float(bins[i + 1])),
        }
    
    return results


def coverage_miss_analysis(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Dict:
    """
    Detailed breakdown of WHERE intervals fail.
    
    Asymmetric misses suggest: your intervals are biased.
    """
    below_mask = y_true < lower
    above_mask = y_true > upper
    hit_mask = ~(below_mask | above_mask)
    
    n_below = np.sum(below_mask)
    n_above = np.sum(above_mask)
    n_hit = np.sum(hit_mask)
    n_total = len(y_true)
    
    if n_below > 0:
        below_error = np.mean(lower[below_mask] - y_true[below_mask])
    else:
        below_error = 0
    
    if n_above > 0:
        above_error = np.mean(y_true[above_mask] - upper[above_mask])
    else:
        above_error = 0
    
    return {
        "total_samples": n_total,
        "hits": {
            "count": n_hit,
            "rate": float(n_hit / n_total),
        },
        "misses_below_lower": {
            "count": n_below,
            "rate": float(n_below / n_total),
            "mean_error": float(below_error) if n_below > 0 else 0,
        },
        "misses_above_upper": {
            "count": n_above,
            "rate": float(n_above / n_total),
            "mean_error": float(above_error) if n_above > 0 else 0,
        },
        "bias": "asymmetric_high" if n_above > n_below else "asymmetric_low" if n_below > n_above else "symmetric",
    }


def print_coverage_analysis(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    datetime_index: Optional[pd.DatetimeIndex] = None,
    horizons: List[int] = [1, 6, 24, 168],
) -> None:
    """Print full diagnostic report."""
    
    print("\n" + "="*70)
    print("COVERAGE BREAKDOWN ANALYSIS")
    print("="*70)
    
    # By horizon
    print("\n1. COVERAGE BY FORECAST HORIZON")
    print("-" * 70)
    horizon_results = coverage_by_horizon(y_true, lower, upper, horizons)
    for h_name, metrics in horizon_results.items():
        print(f"  {h_name:<12} Coverage: {metrics['coverage']:.1%}  Width: {metrics['interval_width']:>8.0f}")
        print(f"             Misses: {metrics['misses_below']:>3} below, {metrics['misses_above']:>3} above")
    
    # By hour
    print("\n2. COVERAGE BY HOUR OF DAY")
    print("-" * 70)
    hour_results = coverage_by_hour_of_day(y_true, lower, upper, datetime_index)
    
    # Group into 4-hour blocks for readability
    for start_hour in range(0, 24, 4):
        hours_in_block = list(range(start_hour, min(start_hour + 4, 24)))
        coverages = [hour_results[h]["coverage"] for h in hours_in_block if h in hour_results]
        
        if coverages:
            mean_cov = np.mean(coverages)
            cov_str = f"{mean_cov:.1%}"
            hour_range = f"{start_hour:02d}-{hours_in_block[-1]:02d}"
            print(f"  Hours {hour_range}:  {cov_str}")
    
    # By magnitude
    print("\n3. COVERAGE BY LOAD MAGNITUDE")
    print("-" * 70)
    mag_results = coverage_by_magnitude(y_true, lower, upper, n_bins=5)
    for bin_label, metrics in mag_results.items():
        print(f"  {bin_label:<25}  Coverage: {metrics['coverage']:.1%}  Width: {metrics['interval_width']:>8.0f}")
    
    # Miss analysis
    print("\n4. MISS ANALYSIS")
    print("-" * 70)
    miss_analysis = coverage_miss_analysis(y_true, lower, upper)
    print(f"  Total samples: {miss_analysis['total_samples']:,}")
    print(f"  Hits:          {miss_analysis['hits']['count']:>6,} ({miss_analysis['hits']['rate']:.1%})")
    print(f"  Misses below:  {miss_analysis['misses_below_lower']['count']:>6,} ({miss_analysis['misses_below_lower']['rate']:.1%})")
    print(f"  Misses above:  {miss_analysis['misses_above_upper']['count']:>6,} ({miss_analysis['misses_above_upper']['rate']:.1%})")
    print(f"  Bias:          {miss_analysis['bias']}")
    
    print("\n" + "="*70 + "\n")
