"""
Model Monitoring & Drift Detection
====================================
Tracks model performance metrics in real-time.
Alerts when coverage drops or RMSE spikes.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class ModelMonitor:
    """Track model performance over time. Alert when things break."""
    
    def __init__(self, metrics_history_path: str = "models/metrics_history.jsonl"):
        self.metrics_path = metrics_history_path
        Path(self.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    
    def log_batch_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        horizon: int = 24,
        stage: str = "test",
    ) -> None:
        """Log metrics for a batch of predictions."""
        
        from src.evaluation.metrics import picp, mpiw, winkler_score
        
        coverage = float(picp(y_true, lower, upper))
        width = float(mpiw(lower, upper))
        winkler = float(winkler_score(y_true, lower, upper, alpha=0.20))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "horizon": horizon,
            "stage": stage,
            "n_samples": int(len(y_true)),
            "coverage": coverage,
            "interval_width": width,
            "winkler_score": winkler,
            "rmse": rmse,
            "mae": mae,
        }
        
        # Append to JSONL
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(record) + "\n")
        
        # Check for alerts
        self._check_alerts(record)
    
    def _check_alerts(self, record: Dict) -> None:
        """Alert if metrics are out of bounds."""
        
        alerts = []
        
        # Coverage thresholds
        if record["coverage"] < 0.75:
            alerts.append(
                f"⚠ ALERT: Coverage dropped to {record['coverage']:.1%} (below 75% threshold)"
            )
        elif record["coverage"] > 0.95:
            alerts.append(
                f"⚠ WARNING: Coverage {record['coverage']:.1%} (too high, intervals may be too wide)"
            )
        
        # Winkler threshold
        if record["winkler_score"] > 200:
            alerts.append(
                f"⚠ ALERT: Winkler score {record['winkler_score']:.0f} (baseline: ~140)"
            )
        
        # RMSE threshold
        baseline_rmse = 1847  # From original evaluation
        if record["rmse"] > baseline_rmse * 1.20:
            alerts.append(
                f"⚠ ALERT: RMSE {record['rmse']:.0f} MW (baseline: {baseline_rmse}, +20% threshold)"
            )
        
        # Compare to rolling average
        if record["stage"] != "train":  # Only check on val/test
            history = self._get_recent_history(days=7)
            if len(history) > 5:  # Need minimum history
                mean_coverage = np.mean([r["coverage"] for r in history])
                if record["coverage"] < mean_coverage - 0.05:
                    alerts.append(
                        f"⚠ TREND: Coverage dropped {(mean_coverage - record['coverage'])*100:.1f} pp "
                        f"from 7-day average ({mean_coverage:.1%})"
                    )
        
        if alerts:
            print("\n" + "="*70)
            for alert in alerts:
                print(alert)
            print("="*70 + "\n")
    
    def _get_recent_history(self, days: int = 7) -> List[Dict]:
        """Load metrics from last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        history = []
        
        if Path(self.metrics_path).exists():
            with open(self.metrics_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    try:
                        ts = datetime.fromisoformat(record["timestamp"])
                        if ts > cutoff:
                            history.append(record)
                    except (ValueError, KeyError):
                        continue
        
        return history
    
    def get_report(self, days: int = 7) -> Dict:
        """Return summary report of N days of metrics."""
        history = self._get_recent_history(days=days)
        
        if not history:
            return {"error": "No metrics history", "n_records": 0}
        
        coverages = [r["coverage"] for r in history]
        winklers = [r["winkler_score"] for r in history]
        rmses = [r["rmse"] for r in history]
        maes = [r["mae"] for r in history]
        
        # Trend: is latest better or worse than mean of history?
        cov_trend = "↓ worse" if coverages[-1] < np.mean(coverages[:-1]) else "↑ better"
        rmse_trend = "↑ worse" if rmses[-1] > np.mean(rmses[:-1]) else "↓ better"
        
        return {
            "period_days": days,
            "n_batches": len(history),
            "coverage": {
                "mean": float(np.mean(coverages)),
                "std": float(np.std(coverages)),
                "min": float(np.min(coverages)),
                "max": float(np.max(coverages)),
                "latest": float(coverages[-1]),
                "trend": cov_trend,
            },
            "winkler_score": {
                "mean": float(np.mean(winklers)),
                "std": float(np.std(winklers)),
                "min": float(np.min(winklers)),
                "max": float(np.max(winklers)),
            },
            "rmse": {
                "mean": float(np.mean(rmses)),
                "std": float(np.std(rmses)),
                "latest": float(rmses[-1]),
                "trend": rmse_trend,
            },
            "mae": {
                "mean": float(np.mean(maes)),
                "latest": float(maes[-1]),
            },
        }
    
    def print_report(self, days: int = 7) -> None:
        """Pretty-print monitoring report."""
        report = self.get_report(days=days)
        
        if "error" in report:
            print(f"No metrics to report: {report['error']}")
            return
        
        print("\n" + "="*70)
        print(f"MONITORING REPORT ({report['period_days']}-day rolling window)")
        print("="*70)
        
        print(f"\nCoverage (target: 80%):")
        print(f"  Mean:   {report['coverage']['mean']:.1%}")
        print(f"  Std:    {report['coverage']['std']:.2%}")
        print(f"  Range:  {report['coverage']['min']:.1%} → {report['coverage']['max']:.1%}")
        print(f"  Latest: {report['coverage']['latest']:.1%} {report['coverage']['trend']}")
        
        print(f"\nWinkler Score:")
        print(f"  Mean:   {report['winkler_score']['mean']:.1f}")
        print(f"  Std:    {report['winkler_score']['std']:.1f}")
        print(f"  Range:  {report['winkler_score']['min']:.1f} → {report['winkler_score']['max']:.1f}")
        
        print(f"\nRMSE (baseline: 1847 MW):")
        print(f"  Mean:   {report['rmse']['mean']:.0f} MW")
        print(f"  Latest: {report['rmse']['latest']:.0f} MW {report['rmse']['trend']}")
        
        print(f"\nMAE:")
        print(f"  Mean:   {report['mae']['mean']:.0f} MW")
        print(f"  Latest: {report['mae']['latest']:.0f} MW")
        
        print(f"\nBatches processed: {report['n_batches']}")
        print("="*70 + "\n")
    
    def clear_history(self) -> None:
        """Clear all metrics history (use cautiously)."""
        if Path(self.metrics_path).exists():
            Path(self.metrics_path).unlink()
            print(f"Cleared metrics history at {self.metrics_path}")
