"""
Conformal Prediction for Time Series
======================================
Two methods:

1. SPLIT CONFORMAL (baseline)
   - Compute nonconformity scores on held-out calibration set
   - Set threshold at (1-alpha) quantile of scores
   - Problem: assumes exchangeability → violated by temporal dependence

2. EnbPI (Ensemble Batch Prediction Intervals) - PREFERRED
   - Rolling calibration: update conformal scores as time progresses
   - Handles distributional shift and temporal dependence
   - Reference: Xu & Xie (2021) "Conformal Prediction Interval for 
     Dynamic Time-Series" ICML 2021

The key distinction interviewers probe: "Why not just use split conformal?"
→ Because time series violate exchangeability (scores are temporally correlated).
→ EnbPI uses a rolling window + online update to maintain coverage.
"""

import numpy as np
from typing import Optional, Tuple


class SplitConformal:
    """
    Standard split conformal prediction.
    
    Provides marginal coverage guarantee:
        P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 - alpha
    
    assuming exchangeability. For time series, use EnbPI instead.
    """
    
    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: miscoverage rate. alpha=0.10 → 90% coverage.
        """
        self.alpha = alpha
        self.q_hat: Optional[float] = None
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
    ) -> "SplitConformal":
        """
        Compute the (1-alpha) quantile of nonconformity scores.
        
        Nonconformity score: s_i = |y_i - ŷ_i|  (residual magnitude)
        
        The conformal quantile q̂ is set so that
        at least ceiling((n+1)(1-alpha))/n fraction of calibration 
        scores are ≤ q̂.
        """
        scores = np.abs(y_cal - y_pred_cal)
        n = len(scores)
        
        # Finite-sample correction: (ceil((n+1)(1-alpha))) / n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        
        self.q_hat = float(np.quantile(scores, level))
        self.calibration_scores = scores
        
        print(f"  Conformal calibration: n={n}, alpha={self.alpha}, q̂={self.q_hat:.2f}")
        return self
    
    def predict_interval(
        self,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct prediction intervals: [ŷ - q̂, ŷ + q̂]
        
        Returns:
            lower: lower bound array
            upper: upper bound array
        """
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() first.")
        
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        
        return lower, upper
    
    def coverage(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """Empirical coverage rate."""
        return float(np.mean((y_true >= lower) & (y_true <= upper)))


class EnbPI:
    """
    Ensemble Batch Prediction Intervals (Xu & Xie, ICML 2021).
    
    Designed for TIME SERIES where exchangeability is violated.
    
    Key idea:
    - Maintain a rolling window of calibration residuals
    - At each new step, update the threshold based on recent errors
    - Handles distribution shift, temporal dependence, concept drift
    
    This is what you tell interviewers when they ask why you didn't
    use standard split conformal on time series.
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        window_size: int = 720,  # ~30 days of hourly data
        beta: float = 0.005,     # Update rate for miscoverage correction
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.beta = beta
        self.residuals: list = []
        self.q_hat: Optional[float] = None
        self.coverage_history: list = []
        self.width_history: list = []
    
    def initialize(
        self,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
    ) -> "EnbPI":
        """Initialize residual buffer from calibration set."""
        scores = np.abs(y_cal - y_pred_cal)
        
        # Seed the rolling window
        self.residuals = list(scores[-self.window_size :])
        self._update_threshold()
        
        print(f"  EnbPI initialized: window={self.window_size}, alpha={self.alpha}, q̂={self.q_hat:.2f}")
        return self
    
    def _update_threshold(self):
        """Recompute q_hat from current rolling residuals."""
        if len(self.residuals) == 0:
            self.q_hat = 0.0
            return
        
        arr = np.array(self.residuals)
        n = len(arr)
        
        # Standard conformal quantile on the rolling window
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        
        self.q_hat = float(np.quantile(arr, level))
    
    def predict_interval(
        self,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals using current q_hat.
        Asymmetric intervals (one-sided expansion) possible with beta correction.
        """
        if self.q_hat is None:
            raise RuntimeError("Call .initialize() first.")
        
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        
        return lower, upper
    
    def update(
        self,
        y_true_new: np.ndarray,
        y_pred_new: np.ndarray,
    ) -> None:
        """
        Online update: add new residuals, drop oldest.
        
        Call this after observing each batch of actuals.
        This is the KEY DIFFERENCE from split conformal —
        we keep calibrating as new data arrives.
        """
        new_scores = np.abs(y_true_new - y_pred_new)
        
        for score in new_scores:
            self.residuals.append(score)
            if len(self.residuals) > self.window_size:
                self.residuals.pop(0)
        
        self._update_threshold()
    
    def rolling_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        batch_size: int = 24,
    ) -> dict:
        """
        Simulate online EnbPI inference over test set.
        Processes in batches; updates after each batch.
        
        Returns dict with coverage, width, and horizon-stratified metrics.
        """
        n = len(y_true)
        all_lower = []
        all_upper = []
        coverages = []
        widths = []
        
        for i in range(0, n, batch_size):
            batch_pred = y_pred[i : i + batch_size]
            batch_true = y_true[i : i + batch_size]
            
            lower, upper = self.predict_interval(batch_pred)
            all_lower.extend(lower)
            all_upper.extend(upper)
            
            # Coverage for this batch
            batch_cov = np.mean((batch_true >= lower) & (batch_true <= upper))
            coverages.append(batch_cov)
            widths.append(np.mean(upper - lower))
            
            # Update with actuals (online calibration)
            self.update(batch_true, batch_pred)
        
        all_lower = np.array(all_lower)
        all_upper = np.array(all_upper)
        
        return {
            "lower": all_lower,
            "upper": all_upper,
            "coverage": float(np.mean((y_true[:n] >= all_lower) & (y_true[:n] <= all_upper))),
            "mean_width": float(np.mean(all_upper - all_lower)),
            "coverage_over_time": coverages,
            "width_over_time": widths,
        }


def evaluate_coverage_by_horizon(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    horizons: list = [1, 6, 24, 168],
) -> dict:
    """
    Stratify coverage and width by forecast horizon.
    
    CRITICAL: Winkler scores are NOT comparable across horizons
    (wider intervals are expected at longer horizons).
    Always stratify by horizon when comparing models.
    """
    n = len(y_true)
    results = {}
    
    for h in horizons:
        indices = list(range(h - 1, n, h))
        if not indices:
            continue
        
        y_h = y_true[indices]
        l_h = lower[indices]
        u_h = upper[indices]
        
        cov = np.mean((y_h >= l_h) & (y_h <= u_h))
        width = np.mean(u_h - l_h)
        
        results[f"h{h}"] = {
            "coverage": round(float(cov), 4),
            "mean_width": round(float(width), 2),
            "n_samples": len(indices),
        }
    
    return results
