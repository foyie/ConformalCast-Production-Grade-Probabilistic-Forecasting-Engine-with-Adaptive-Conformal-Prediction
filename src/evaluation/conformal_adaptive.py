"""
Adaptive EnbPI with Drift Detection
====================================
Extends EnbPI to detect distribution shifts and adapt window size.
Solves the problem: "Fixed window gets stale when data drifts"
"""

import numpy as np
from typing import Tuple, Dict
from scipy.stats import ks_2samp
from src.evaluation.conformal import EnbPI


class AdaptiveEnbPI(EnbPI):
    """
    EnbPI that detects distribution drift and adjusts calibration window.
    
    If recent predictions have different error distribution than history,
    shrink the window to be more responsive. Else slowly expand back to normal.
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        initial_window: int = 720,
        min_window: int = 240,
        max_window: int = 1440,
        drift_threshold: float = 0.05,
    ):
        """
        Args:
            alpha: miscoverage rate (0.10 → 90% nominal coverage)
            initial_window: hours for rolling calibration
            min_window: shrink window to at most this small (24 hours min)
            max_window: grow window to at most this large
            drift_threshold: p-value for KS test to flag drift (p < threshold = drift)
        """
        super().__init__(alpha=alpha, window_size=initial_window)
        self.min_window = min_window
        self.max_window = max_window
        self.drift_threshold = drift_threshold
        self.drift_history: list = []
        self.window_history: list = []
    
    def _detect_drift(self) -> Tuple[bool, float, float]:
        """
        Kolmogorov-Smirnov test for distribution shift.
        
        Returns:
            (drift_detected, ks_stat, p_value)
        
        Theory: If recent residuals have different distribution than historical,
        the test rejects equality hypothesis (p < 0.05).
        """
        if len(self.residuals) < self.window_size + 100:
            # Not enough data for reliable drift detection
            return False, 0.0, 1.0
        
        # Recent vs historical
        recent = np.array(self.residuals[-240:])  # Last 10 days of hourly data
        historical = np.array(self.residuals[:-240])
        
        if len(recent) < 50 or len(historical) < 50:
            return False, 0.0, 1.0
        
        # KS test: are these distributions the same?
        ks_stat, p_value = ks_2samp(recent, historical)
        
        # Drift detected if p < threshold (reject null hypothesis of equality)
        drift_detected = p_value < self.drift_threshold
        
        return drift_detected, float(ks_stat), float(p_value)
    
    def update(
        self,
        y_true_new: np.ndarray,
        y_pred_new: np.ndarray,
    ) -> None:
        """
        Update with new batch. Detect drift and adapt window.
        """
        # Parent class: add residuals and update threshold
        super().update(y_true_new, y_pred_new)
        
        # Check for drift
        drift_detected, ks_stat, p_value = self._detect_drift()
        self.drift_history.append({
            "drift": drift_detected,
            "ks_stat": ks_stat,
            "p_value": p_value,
            "window_before": self.window_size,
        })
        
        # Adapt window
        if drift_detected:
            # Shrink to be more responsive
            new_window = max(self.min_window, int(self.window_size * 0.80))
            print(f"  ⚠ DRIFT DETECTED (p={p_value:.4f}). Shrinking window: {self.window_size}h → {new_window}h")
        else:
            # Gradually expand back to nominal
            new_window = min(self.max_window, int(self.window_size * 1.01))
        
        self.window_size = new_window
        self.window_history.append(self.window_size)
    
    def rolling_coverage_adaptive(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        batch_size: int = 24,
    ) -> Dict:
        """
        Simulate deployment: process batches, update with drift detection.
        
        Returns: dict with coverage, width, drift events
        """
        n = len(y_true)
        all_lower = []
        all_upper = []
        coverages = []
        widths = []
        drift_events = []
        
        for i in range(0, n, batch_size):
            batch_pred = y_pred[i : i + batch_size]
            batch_true = y_true[i : i + batch_size]
            
            # Generate intervals
            lower, upper = self.predict_interval(batch_pred)
            all_lower.extend(lower)
            all_upper.extend(upper)
            
            # Coverage for this batch
            batch_cov = np.mean((batch_true >= lower) & (batch_true <= upper))
            coverages.append(batch_cov)
            widths.append(np.mean(upper - lower))
            
            # Update with drift detection
            self.update(batch_true, batch_pred)
            
            # Log drift events
            if self.drift_history:
                last_drift = self.drift_history[-1]
                if last_drift["drift"]:
                    drift_events.append({
                        "batch": i // batch_size,
                        "ks_stat": last_drift["ks_stat"],
                        "p_value": last_drift["p_value"],
                    })
        
        all_lower = np.array(all_lower)
        all_upper = np.array(all_upper)
        
        overall_coverage = float(np.mean((y_true[:n] >= all_lower) & (y_true[:n] <= all_upper)))
        
        return {
            "lower": all_lower,
            "upper": all_upper,
            "coverage": overall_coverage,
            "mean_width": float(np.mean(all_upper - all_lower)),
            "coverage_over_time": coverages,
            "width_over_time": widths,
            "drift_events": drift_events,
            "n_drift_events": len(drift_events),
            "final_window": self.window_size,
        }
    
    def get_drift_report(self) -> Dict:
        """Return summary of drift detection."""
        if not self.drift_history:
            return {"status": "No drift events recorded"}
        
        drifts = np.array([d["drift"] for d in self.drift_history])
        n_drifts = np.sum(drifts)
        
        return {
            "total_updates": len(self.drift_history),
            "drift_events": int(n_drifts),
            "drift_rate": float(n_drifts / len(self.drift_history)),
            "window_range": (min(self.window_history), max(self.window_history)),
            "window_mean": float(np.mean(self.window_history)),
        }
