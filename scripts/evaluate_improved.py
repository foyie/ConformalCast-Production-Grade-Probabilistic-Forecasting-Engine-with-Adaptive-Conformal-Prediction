"""
Evaluation Pipeline (Updated)
==============================
Now includes monitoring, drift detection, and coverage breakdown.

Run: python scripts/evaluate.py
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_artifacts():
    """Load trained models and calibrators."""
    artifacts = {}

    try:
        from src.models.lgbm_quantile import LGBMQuantileForecaster
        artifacts["lgbm"] = LGBMQuantileForecaster.load("models/lgbm/")
        print("✓ LightGBM loaded")
    except Exception as e:
        print(f"✗ LightGBM: {e}")

    try:
        # Try new adaptive version first
        from src.evaluation.conformal_adaptive import AdaptiveEnbPI
        artifacts["enbpi_95_adaptive"] = joblib.load("models/conformal/enbpi_95.pkl")
        print("✓ Conformal calibrators loaded (will use adaptive version)")
    except Exception as e:
        print(f"  Note: Using standard EnbPI (adaptive version not yet saved)")
        try:
            from src.evaluation.conformal import EnbPI
            artifacts["enbpi_95"] = joblib.load("models/conformal/enbpi_95.pkl")
            artifacts["enbpi_80"] = joblib.load("models/conformal/enbpi_80.pkl")
            print("✓ Standard EnbPI loaded")
        except Exception as e2:
            print(f"✗ Conformal: {e2}")

    return artifacts

def run_evaluation(artifacts: dict, test_df: pd.DataFrame, config: dict) -> dict:
    """Full evaluation pass on test set."""
    from src.utils.features import get_feature_columns
    from src.evaluation.metrics import full_evaluation_report
    from src.evaluation.conformal_adaptive import AdaptiveEnbPI
    from src.serving.monitoring import ModelMonitor
    from src.evaluation.coverage_analysis import print_coverage_analysis

    target_col = config["data"]["target_col"]
    feature_cols = get_feature_columns(test_df, target_col)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values
    datetime_index = test_df["Datetime"].values

    # LightGBM quantile predictions
    lgbm = artifacts["lgbm"]
    quantile_preds = lgbm.predict(X_test)
    y_pred = quantile_preds[0.50]

    adaptive_mode = False
    eval_start = 0

    # ── Try adaptive conformal first ─────────────────────────────────────────
    try:
        adaptive_mode = True
        eval_start = len(y_test) // 2

        # 95% adaptive intervals
        enbpi_95_adaptive = AdaptiveEnbPI(alpha=0.10, initial_window=720)
        enbpi_95_adaptive.initialize(
            y_test[:eval_start],
            y_pred[:eval_start]
        )

        result_95 = enbpi_95_adaptive.rolling_coverage_adaptive(
            y_test[eval_start:],
            y_pred[eval_start:],
            batch_size=24
        )

        # 80% adaptive intervals
        enbpi_80_adaptive = AdaptiveEnbPI(alpha=0.20, initial_window=720)
        enbpi_80_adaptive.initialize(
            y_test[:eval_start],
            y_pred[:eval_start]
        )

        result_80 = enbpi_80_adaptive.rolling_coverage_adaptive(
            y_test[eval_start:],
            y_pred[eval_start:],
            batch_size=24
        )

        print("\n✓ Using ADAPTIVE EnbPI with drift detection")

        # Log drift events
        if result_95.get("n_drift_events", 0) > 0:
            print(f"  Detected {result_95['n_drift_events']} drift events")
            print(f"  Final window: {result_95['final_window']}h (adjusted from 720h)")

    # ── Fallback to standard conformal ───────────────────────────────────────
    except Exception as e:
        adaptive_mode = False
        eval_start = 0

        print(f"\n✓ Using standard EnbPI: {e}")

        enbpi_95 = artifacts.get("enbpi_95") or joblib.load(
            "models/conformal/enbpi_95.pkl"
        )
        enbpi_80 = artifacts.get("enbpi_80") or joblib.load(
            "models/conformal/enbpi_80.pkl"
        )

        result_95 = enbpi_95.rolling_coverage(
            y_test,
            y_pred,
            batch_size=24
        )

        result_80 = enbpi_80.rolling_coverage(
            y_test,
            y_pred,
            batch_size=24
        )

    # ── Extract intervals ────────────────────────────────────────────────────
    lower_95 = result_95["lower"]
    upper_95 = result_95["upper"]

    lower_80 = result_80["lower"]
    upper_80 = result_80["upper"]

    # ── Slice evaluation set consistently ────────────────────────────────────
    y_test_eval = y_test[eval_start:]
    y_pred_eval = y_pred[eval_start:]
    datetime_eval = datetime_index[eval_start:]

    quantile_preds_eval = {
        q: preds[eval_start:]
        for q, preds in quantile_preds.items()
    }

    # Sanity check
    assert len(y_test_eval) == len(lower_80) == len(lower_95), (
        f"Shape mismatch:\n"
        f"y_test_eval={len(y_test_eval)}\n"
        f"lower_80={len(lower_80)}\n"
        f"lower_95={len(lower_95)}"
    )

    # ── Full evaluation report ───────────────────────────────────────────────
    report = full_evaluation_report(
        y_true=y_test_eval,
        y_pred=y_pred_eval,
        lower_80=lower_80,
        upper_80=upper_80,
        lower_95=lower_95,
        upper_95=upper_95,
        quantile_preds=quantile_preds_eval,
    )

    # ── Rolling coverage diagnostics ─────────────────────────────────────────
    report["coverage_drift"] = {
        "coverage_95_over_time": result_95["coverage_over_time"],
        "coverage_80_over_time": result_80["coverage_over_time"],
        "width_over_time": result_95["width_over_time"],
    }

    # Adaptive metadata
    if adaptive_mode:
        report["adaptive_conformal"] = {
            "enabled": True,
            "initial_window": 720,
            "final_window": result_95.get("final_window", 720),
            "n_drift_events": result_95.get("n_drift_events", 0),
        }
    else:
        report["adaptive_conformal"] = {
            "enabled": False
        }

    report["evaluated_at"] = datetime.utcnow().isoformat()
    report["n_test_samples"] = len(y_test_eval)
    report["calibration_samples"] = eval_start if adaptive_mode else 0

    # ── Coverage breakdown analysis ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: COVERAGE BREAKDOWN")
    print("=" * 70)

    print_coverage_analysis(
        y_test_eval,
        lower_80,
        upper_80,
        datetime_index=pd.to_datetime(datetime_eval),
        horizons=[1, 6, 24, 168],
    )

    # ── Log to monitoring system ─────────────────────────────────────────────
    print("=" * 70)
    print("LOGGING TO MONITORING SYSTEM")
    print("=" * 70)

    monitor = ModelMonitor()
    monitor.log_batch_metrics(
        y_test_eval,
        y_pred_eval,
        lower_80,
        upper_80,
        horizon=24,
        stage="test"
    )

    monitor.print_report(days=1)

    return report

# def run_evaluation(artifacts: dict, test_df: pd.DataFrame, config: dict) -> dict:
#     """Full evaluation pass on test set."""
#     from src.utils.features import get_feature_columns
#     from src.evaluation.metrics import full_evaluation_report
#     from src.evaluation.conformal_adaptive import AdaptiveEnbPI
#     from src.serving.monitoring import ModelMonitor
#     from src.evaluation.coverage_analysis import print_coverage_analysis

#     target_col = config["data"]["target_col"]
#     feature_cols = get_feature_columns(test_df, target_col)

#     X_test = test_df[feature_cols]
#     y_test = test_df[target_col].values
#     datetime_index = test_df["Datetime"].values

#     # LightGBM quantile predictions
#     lgbm = artifacts["lgbm"]
#     quantile_preds = lgbm.predict(X_test)
#     y_pred = quantile_preds[0.50]

#     # Try adaptive conformal first
#     try:
#         enbpi_95_adaptive = AdaptiveEnbPI(alpha=0.10, initial_window=720)
#         enbpi_95_adaptive.initialize(y_test[:len(y_test)//2], y_pred[:len(y_pred)//2])
#         result_95 = enbpi_95_adaptive.rolling_coverage_adaptive(
#             y_test[len(y_test)//2:],
#             y_pred[len(y_pred)//2:],
#             batch_size=24
#         )
#         print("\n✓ Using ADAPTIVE EnbPI with drift detection")

#         # Log drift events
#         if result_95["n_drift_events"] > 0:
#             print(f"  Detected {result_95['n_drift_events']} drift events")
#             print(f"  Final window: {result_95['final_window']}h (adjusted from 720h)")

#     except Exception as e:
#         print(f"\n✓ Using standard EnbPI: {e}")
#         from src.evaluation.conformal import EnbPI

#         enbpi_95 = artifacts.get("enbpi_95") or joblib.load("models/conformal/enbpi_95.pkl")
#         enbpi_80 = artifacts.get("enbpi_80") or joblib.load("models/conformal/enbpi_80.pkl")

#         result_95 = enbpi_95.rolling_coverage(y_test, y_pred, batch_size=24)
#         result_80 = enbpi_80.rolling_coverage(y_test, y_pred, batch_size=24)

#         lower_80 = result_80["lower"]
#         upper_80 = result_80["upper"]

#     # Get intervals
#     lower_95 = result_95["lower"]
#     upper_95 = result_95["upper"]

#     if "lower" not in result_95:
#         # Standard EnbPI returned, get 80% from separate result
#         lower_80 = result_80["lower"]
#         upper_80 = result_80["upper"]
#     else:
#         # Adaptive version, convert 95% to 80%
#         from src.evaluation.conformal_adaptive import AdaptiveEnbPI
#         enbpi_80_adaptive = AdaptiveEnbPI(alpha=0.20, initial_window=720)
#         enbpi_80_adaptive.initialize(y_test[:len(y_test)//2], y_pred[:len(y_pred)//2])
#         result_80 = enbpi_80_adaptive.rolling_coverage_adaptive(
#             y_test[len(y_test)//2:],
#             y_pred[len(y_pred)//2:],
#             batch_size=24
#         )
#         lower_80 = result_80["lower"]
#         upper_80 = result_80["upper"]

#     # Full report
#     report = full_evaluation_report(
#         y_true=y_test,
#         y_pred=y_pred,
#         lower_80=lower_80,
#         upper_80=upper_80,
#         lower_95=lower_95,
#         upper_95=upper_95,
#         quantile_preds=quantile_preds,
#     )

#     # Rolling coverage over time
#     report["coverage_drift"] = {
#         "coverage_95_over_time": result_95["coverage_over_time"],
#         "coverage_80_over_time": result_80["coverage_over_time"],
#         "width_over_time": result_95["width_over_time"],
#     }

#     report["evaluated_at"] = datetime.utcnow().isoformat()
#     report["n_test_samples"] = len(y_test)

#     # ── Coverage breakdown analysis ──────────────────────────────────────────
#     print("\n" + "="*70)
#     print("DIAGNOSTIC: COVERAGE BREAKDOWN")
#     print("="*70)
#     print_coverage_analysis(
#         y_test,
#         lower_80,
#         upper_80,
#         datetime_index=pd.to_datetime(datetime_index),
#         horizons=[1, 6, 24, 168],
#     )

#     # ── Log to monitoring system ─────────────────────────────────────────────
#     print("="*70)
#     print("LOGGING TO MONITORING SYSTEM")
#     print("="*70)
#     monitor = ModelMonitor()
#     monitor.log_batch_metrics(y_test, y_pred, lower_80, upper_80, horizon=24, stage="test")
#     monitor.print_report(days=1)

#     return report


def print_summary(report: dict) -> None:
    """Pretty-print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    ov = report["overall"]
    print(f"\nPoint forecast:")
    print(f"  RMSE : {ov['rmse']:>10,.1f} MW")
    print(f"  MAE  : {ov['mae']:>10,.1f} MW")
    print(f"  MAPE : {ov['mape']:>10.2f}%")

    i80 = report["interval_80"]
    print(f"\n80% prediction intervals:")
    print(f"  Nominal coverage  : {i80['nominal_coverage']:.0%}")
    print(f"  Empirical coverage: {i80['picp']:.1%}  ← should be ≥80%")
    print(f"  Winkler score     : {i80['winkler']:>10,.1f}  (lower=better)")
    print(f"  Mean width        : {i80['mpiw']:>10,.1f} MW")

    i95 = report["interval_95"]
    print(f"\n95% prediction intervals:")
    print(f"  Nominal coverage  : {i95['nominal_coverage']:.0%}")
    print(f"  Empirical coverage: {i95['picp']:.1%}  ← should be ≥95%")
    print(f"  Winkler score     : {i95['winkler']:>10,.1f}")
    print(f"  Mean width        : {i95['mpiw']:>10,.1f} MW")

    if "calibration" in report:
        cal = report["calibration"]
        print(f"\nCalibration (reliability diagram):")
        print(f"  ECE: {cal['ece']:.4f}  (0 = perfect)")

    print(f"\nBy horizon (80% intervals):")
    for h_key, h_metrics in report["horizon_metrics_80"].items():
        print(f"  {h_key:<8}: PICP={h_metrics['picp']:.1%}  "
              f"Winkler={h_metrics['winkler']:>8,.0f}  "
              f"Width={h_metrics['mpiw']:>8,.0f}")

    print("="*60)


def main():
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    print("Loading artifacts...")
    artifacts = load_artifacts()

    if not artifacts:
        print("No artifacts found. Run scripts/train.py first.")
        sys.exit(1)

    print("Loading test set...")
    test_df = pd.read_csv("data/processed/test_set.csv")
    test_df["Datetime"] = pd.to_datetime(test_df["Datetime"])
    print(f"Test set: {len(test_df):,} rows")

    print("Running evaluation...")
    report = run_evaluation(artifacts, test_df, config)

    print_summary(report)

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Results saved to results/metrics.json")
    print("Open dashboard/index.html to visualize results")


if __name__ == "__main__":
    main()
