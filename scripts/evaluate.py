"""
Evaluation Pipeline
====================
Loads test set, runs inference, computes all metrics:
  - Winkler scores by horizon
  - PICP @ 80%, 90%, 95%
  - Reliability diagram data
  - ECE (Expected Calibration Error)
  - Coverage drift over time

Output: results/metrics.json, results/plots/
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
        artifacts["enbpi_95"] = joblib.load("models/conformal/enbpi_95.pkl")
        artifacts["enbpi_80"] = joblib.load("models/conformal/enbpi_80.pkl")
        print("✓ Conformal calibrators loaded")
    except Exception as e:
        print(f"✗ Conformal: {e}")
    
    return artifacts


def run_evaluation(artifacts: dict, test_df: pd.DataFrame, config: dict) -> dict:
    """Full evaluation pass on test set."""
    from src.utils.features import get_feature_columns
    from src.evaluation.metrics import full_evaluation_report, horizon_metrics
    from src.evaluation.conformal import evaluate_coverage_by_horizon
    
    target_col = config["data"]["target_col"]
    feature_cols = get_feature_columns(test_df, target_col)
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values
    
    # LightGBM quantile predictions
    lgbm = artifacts["lgbm"]
    quantile_preds = lgbm.predict(X_test)
    y_pred = quantile_preds[0.50]
    
    # Conformal intervals
    enbpi_95 = artifacts["enbpi_95"]
    enbpi_80 = artifacts["enbpi_80"]
    
    result_95 = enbpi_95.rolling_coverage(y_test, y_pred, batch_size=24)
    result_80 = enbpi_80.rolling_coverage(y_test, y_pred, batch_size=24)
    
    lower_95 = result_95["lower"]
    upper_95 = result_95["upper"]
    lower_80 = result_80["lower"]
    upper_80 = result_80["upper"]
    
    # Full report
    report = full_evaluation_report(
        y_true=y_test,
        y_pred=y_pred,
        lower_80=lower_80,
        upper_80=upper_80,
        lower_95=lower_95,
        upper_95=upper_95,
        quantile_preds=quantile_preds,
    )
    
    # Rolling coverage over time
    report["coverage_drift"] = {
        "coverage_95_over_time": result_95["coverage_over_time"],
        "coverage_80_over_time": result_80["coverage_over_time"],
        "width_over_time": result_95["width_over_time"],
    }
    
    report["evaluated_at"] = datetime.utcnow().isoformat()
    report["n_test_samples"] = len(y_test)
    
    return report


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
        print(f"  {'Quantile':>10}  {'Nominal':>10}  {'Empirical':>10}  {'Error':>8}")
        for nom, emp in zip(cal["nominal"], cal["empirical"]):
            print(f"  {nom:>10.2f}  {nom:>10.1%}  {emp:>10.1%}  {abs(nom-emp):>8.4f}")
    
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
    
    print(f"\nResults saved to results/metrics.json")
    print("Open dashboard/index.html to visualize results")


if __name__ == "__main__":
    main()
