"""
LightGBM Quantile Regression
==============================
Trains separate models for q10, q50, q90 (and optionally q05, q95).
Quantile regression gives asymmetric intervals that adapt to the
local distribution — unlike symmetric Gaussian intervals.

Key insight: train a SEPARATE model per quantile. Joint training
sometimes violates quantile crossing, but separate models are faster
and more interpretable.
"""

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LGBMQuantileForecaster:
    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.10, 0.50, 0.90, 0.95],
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = 1,  # Changed from -1 (all cores) to 1 (single thread) — fixes macOS segfault
    ):
        self.quantiles = quantiles
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: Optional[List[str]] = None
        self.model_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "n_jobs": n_jobs,
            "random_state": 42,
            "verbose": -1,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LGBMQuantileForecaster":
        """
        Train one model per quantile.
        Uses early stopping on validation set when provided.
        """
        self.feature_names = list(X_train.columns)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        for q in self.quantiles:
            print(f"  Training quantile q={q:.2f}...")

            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **self.model_params,
            )

            callbacks = []
            if eval_set is not None:
                callbacks.append(lgb.early_stopping(50, verbose=False))
                callbacks.append(lgb.log_evaluation(period=-1))

            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                eval_metric="quantile",
                callbacks=callbacks if callbacks else None,
            )

            self.models[q] = model

        print(f"  Trained {len(self.models)} quantile models")
        return self

    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Generate quantile predictions.
        Returns dict: {0.10: array, 0.50: array, 0.90: array, ...}
        """
        if not self.models:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        predictions = {}
        for q, model in self.models.items():
            pred = model.predict(X)
            predictions[q] = pred

        # Fix quantile crossing (monotone constraint post-hoc)
        predictions = self._fix_crossing(predictions)

        return predictions

    def predict_interval(
        self, X: pd.DataFrame, coverage: float = 0.80
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (lower, point_forecast, upper) for a given coverage level.
        For 80% coverage: lower=q10, point=q50, upper=q90.
        """
        alpha = (1 - coverage) / 2
        lower_q = round(alpha, 3)
        upper_q = round(1 - alpha, 3)

        if lower_q not in self.models or upper_q not in self.models:
            available = sorted(self.models.keys())
            raise ValueError(
                f"Coverage {coverage} requires q={lower_q} and q={upper_q}. "
                f"Available quantiles: {available}"
            )

        preds = self.predict(X)
        return preds[lower_q], preds[0.50], preds[upper_q]

    def _fix_crossing(self, predictions: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
        """
        Isotonic regression to fix quantile crossing.
        Ensures q10 <= q50 <= q90 element-wise.
        """
        from sklearn.isotonic import IsotonicRegression

        sorted_qs = sorted(predictions.keys())
        n = len(predictions[sorted_qs[0]])

        # Stack quantiles and apply isotonic regression per sample
        matrix = np.stack([predictions[q] for q in sorted_qs], axis=1)

        ir = IsotonicRegression(increasing=True)
        corrected = np.apply_along_axis(
            lambda row: ir.fit_transform(range(len(row)), row),
            axis=1,
            arr=matrix,
        )

        return {q: corrected[:, i] for i, q in enumerate(sorted_qs)}

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return feature importance from median model (q50)."""
        if 0.50 not in self.models:
            model = list(self.models.values())[0]
        else:
            model = self.models[0.50]

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)

        return importance

    def save(self, path: str) -> None:
        """Save all quantile models."""
        Path(path).mkdir(parents=True, exist_ok=True)
        for q, model in self.models.items():
            joblib.dump(model, f"{path}/lgbm_q{int(q*100):02d}.pkl")
        joblib.dump(self.feature_names, f"{path}/feature_names.pkl")
        print(f"Saved {len(self.models)} quantile models to {path}/")

    @classmethod
    def load(cls, path: str) -> "LGBMQuantileForecaster":
        """Load saved quantile models."""
        import glob

        forecaster = cls()
        forecaster.feature_names = joblib.load(f"{path}/feature_names.pkl")

        for fpath in glob.glob(f"{path}/lgbm_q*.pkl"):
            fname = os.path.basename(fpath)
            # Parse q from filename: lgbm_q10.pkl → 0.10
            q_int = int(fname.replace("lgbm_q", "").replace(".pkl", ""))
            q = q_int / 100
            forecaster.models[q] = joblib.load(fpath)

        print(f"Loaded {len(forecaster.models)} quantile models from {path}/")
        return forecaster


def calibration_error(
    y_true: np.ndarray,
    quantile_preds: Dict[float, np.ndarray],
) -> Dict[float, float]:
    """
    Expected Calibration Error (ECE) per quantile.
    ECE = |empirical_coverage - nominal_coverage|
    Well-calibrated model → ECE close to 0 for all quantiles.
    """
    ece = {}
    for q, preds in quantile_preds.items():
        empirical = np.mean(y_true <= preds)
        ece[q] = abs(empirical - q)
    return ece
