"""
Model Monitoring
=================
Track forecast quality in production. Alert on degradation.
Generates JSON log of all predictions for alerting and debugging.

Usage:
  monitor = ModelMonitor()
  monitor.log_batch_metrics(y_true, y_pred, lower, upper, horizon=24)
  report = monitor.get_report()  # 7-day summary
  health = monitor.health_check()  # Quick health status
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class ModelMonitor:
    """Production monitoring for probabilistic forecasts."""

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
        model_name: str = "ensemble",
    ) -> None:
        """
        Log metrics for a batch of predictions.

        Args:
            y_true: actual values
            y_pred: point forecasts
            lower: lower prediction intervals
            upper: upper prediction intervals
            horizon: forecast horizon (hours)
            model_name: which model variant
        """
        from src.evaluation.metrics import picp, mpiw, winkler_score

        # Compute metrics
        coverage = picp(y_true, lower, upper)
        width = mpiw(lower, upper)
        winkler = winkler_score(y_true, lower, upper, alpha=0.20)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # Count misses
        upper_misses = int(np.sum(y_true > upper))
        lower_misses = int(np.sum(y_true < lower))

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "horizon_hours": horizon,
            "model_name": model_name,
            "n_samples": int(len(y_true)),
            "coverage": float(coverage),
            "interval_width": float(width),
            "winkler_score": float(winkler),
            "rmse": float(rmse),
            "mae": float(mae),
            "upper_misses": upper_misses,
            "lower_misses": lower_misses,
            "asymmetry": float(lower_misses - upper_misses) / len(y_true),
        }

        # Append to JSONL
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

        # Check for alerts
        self._check_alerts(record)

    def _check_alerts(self, record: Dict) -> None:
        """Alert if metrics go out of bounds."""

        alerts = []

        if record["coverage"] < 0.75:
            alerts.append(
                f"CRITICAL: Coverage {record['coverage']:.1%} below 75% threshold"
            )
        elif record["coverage"] < 0.78:
            alerts.append(
                f"WARNING: Coverage {record['coverage']:.1%} below 80% target"
            )

        if record["winkler_score"] > 200:
            alerts.append(
                f"WARNING: Winkler score {record['winkler_score']:.0f} exceeds baseline 142"
            )

        if record["rmse"] > 2032:
            alerts.append(
                f"WARNING: RMSE {record['rmse']:.0f} exceeds baseline 1847"
            )

        if abs(record["asymmetry"]) > 0.05:
            alerts.append(
                f"WARNING: Asymmetric misses detected (ratio={record['asymmetry']:.3f})"
            )

        for alert in alerts:
            print(alert)

    def _get_recent_history(self, days: int = 7) -> List[Dict]:
        """Load metrics from last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        history = []

        if Path(self.metrics_path).exists():
            with open(self.metrics_path) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        ts = datetime.fromisoformat(record["timestamp"])
                        if ts > cutoff:
                            history.append(record)
                    except (json.JSONDecodeError, KeyError):
                        continue

        return history

    def get_report(self) -> Dict:
        """Generate 7-day performance summary for dashboard."""
        history = self._get_recent_history(days=7)

        if not history:
            return {"status": "no_data", "message": "No metrics history found"}

        coverages = np.array([r["coverage"] for r in history])
        winklers = np.array([r["winkler_score"] for r in history])
        rmses = np.array([r["rmse"] for r in history])
        widths = np.array([r["interval_width"] for r in history])

        coverage_trend = "Declining" if coverages[-1] < np.mean(coverages[:-1]) else "Improving"
        rmse_trend = "Degrading" if rmses[-1] > np.mean(rmses[:-1]) else "Improving"

        if np.mean(coverages) >= 0.80 and np.mean(coverages) <= 0.95:
            status = "Healthy"
        elif np.mean(coverages) >= 0.75:
            status = "Degraded"
        else:
            status = "Critical"

        return {
            "status": status,
            "period_days": 7,
            "n_batches": len(history),
            "coverage": {
                "mean": float(np.mean(coverages)),
                "std": float(np.std(coverages)),
                "min": float(np.min(coverages)),
                "max": float(np.max(coverages)),
                "trend": coverage_trend,
                "target": 0.80,
            },
            "winkler_score": {
                "mean": float(np.mean(winklers)),
                "std": float(np.std(winklers)),
                "baseline": 142.3,
            },
            "rmse": {
                "mean": float(np.mean(rmses)),
                "trend": rmse_trend,
                "baseline": 1847,
            },
            "interval_width": {
                "mean": float(np.mean(widths)),
                "trend": "Narrowing" if widths[-1] < np.mean(widths[:-1]) else "Widening",
            },
            "recent_records": history[-5:],
        }

    def health_check(self) -> Dict:
        """Quick health check for API /health endpoint."""
        report = self.get_report()

        if report.get("status") == "no_data":
            return {"monitoring": False, "reason": "No metrics history"}

        coverage = report["coverage"]["mean"]
        rmse = report["rmse"]["mean"]
        baseline_rmse = report["rmse"]["baseline"]

        return {
            "monitoring": True,
            "coverage_ok": coverage >= 0.75,
            "rmse_ok": rmse <= baseline_rmse * 1.15,
            "overall_healthy": coverage >= 0.75 and rmse <= baseline_rmse * 1.15,
            "coverage": float(coverage),
            "rmse": float(rmse),
            "status": report["status"],
        }


def print_monitoring_report(monitor: ModelMonitor) -> None:
    """Pretty-print monitoring report (for CLI use)."""
    report = monitor.get_report()

    if report.get("status") == "no_data":
        print("No monitoring data yet")
        return

    print("\n" + "=" * 70)
    print(f"MONITORING REPORT (Last 7 days) - Status: {report['status']}")
    print("=" * 70)

    cov = report["coverage"]
    print(f"\nCOVERAGE (target: 80%)")
    print(f"  Mean:    {cov['mean']:.1%}  [{cov['trend']}]")
    print(f"  Range:   {cov['min']:.1%} to {cov['max']:.1%}")
    print(f"  StdDev:  {cov['std']:.2%}")

    rmse = report["rmse"]
    print(f"\nRMSE (baseline: {rmse['baseline']:.0f} MW)")
    print(f"  Mean:   {rmse['mean']:.0f} MW ({rmse['mean']/rmse['baseline']*100:.0f}% of baseline)")
    print(f"  Trend:  {rmse['trend']}")

    winkler = report["winkler_score"]
    print(f"\nWINKLER SCORE (baseline: {winkler['baseline']:.0f})")
    print(f"  Mean:   {winkler['mean']:.0f}  (std_dev={winkler['std']:.0f})")

    width = report["interval_width"]
    print(f"\nINTERVAL WIDTH")
    print(f"  Mean:   {width['mean']:.0f} MW")
    print(f"  Trend:  {width['trend']}")

    print(f"\nBatches processed: {report['n_batches']}")
    print("=" * 70 + "\n")
