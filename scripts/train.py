

"""
Full Training Pipeline
========================
Trains all models in order:
  1. NeuralProphet (trend + seasonality baseline)
  2. LSTM with MC Dropout (nonlinear temporal patterns)
  3. LightGBM Quantile Regression (tabular lag features)

Run: python scripts/train.py --config configs/config.yaml
"""

import sys
import os
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    validated_path = Path(config["data"]["processed_path"]) / "validated.csv"

    if not validated_path.exists():
        print("Data not found. Running download script...")
        os.system("python scripts/download_data.py")

    df = pd.read_csv(validated_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    print(f"Loaded {len(df):,} rows")
    return df


def train_neuralprophet(df: pd.DataFrame, config: dict) -> None:
    """Train NeuralProphet baseline."""
    print("\n" + "="*50)
    print("STEP 1: NeuralProphet")
    print("="*50)

    try:
        from neuralprophet import NeuralProphet
    except ImportError:
        print("  neuralprophet not installed. Skipping.")
        return

    np_config = config["neuralprophet"]

    # NeuralProphet expects 'ds' and 'y' columns
    np_df = df[["Datetime", config["data"]["target_col"]]].copy()
    np_df.columns = ["ds", "y"]

    # Temporal split
    n = len(np_df)
    train_end = int(n * config["data"]["train_ratio"])

    np_train = np_df.iloc[:train_end]

    model = NeuralProphet(
        n_forecasts=np_config["n_forecasts"],
        n_lags=np_config["n_lags"],
        yearly_seasonality=np_config["yearly_seasonality"],
        weekly_seasonality=np_config["weekly_seasonality"],
        daily_seasonality=np_config["daily_seasonality"],
        learning_rate=np_config["learning_rate"],
        epochs=np_config["epochs"],
        batch_size=np_config["batch_size"],
    )

    metrics = model.fit(np_train, freq="H")

    # Save — NeuralProphet uses a module-level save function, not model.save()
    Path("models/neuralprophet").mkdir(parents=True, exist_ok=True)
    try:
        from neuralprophet import save as np_save
        np_save(model, "models/neuralprophet/model.pkl")
    except ImportError:
        # Older versions: serialize manually
        import pickle
        with open("models/neuralprophet/model.pkl", "wb") as f:
            pickle.dump(model, f)
    print("  ✓ NeuralProphet trained and saved")
    return model


def train_lstm(df: pd.DataFrame, config: dict, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Train LSTM with MC Dropout."""
    print("\n" + "="*50)
    print("STEP 2: LSTM with MC Dropout")
    print("="*50)

    from src.models.lstm_model import MCDropoutForecaster

    lstm_config = config["lstm"]
    input_size = X_train.shape[1]

    print(f"  Input features: {input_size}")
    print(f"  Sequence length: {lstm_config['sequence_length']}h")

    forecaster = MCDropoutForecaster(
        input_size=input_size,
        sequence_length=lstm_config["sequence_length"],
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"],
        dropout=lstm_config["dropout"],
        learning_rate=lstm_config["learning_rate"],
        epochs=lstm_config["epochs"],
        batch_size=lstm_config["batch_size"],
        mc_samples=lstm_config["mc_samples"],
    )

    forecaster.fit(X_train, y_train, X_val, y_val)

    # Save with input_size in metadata
    import joblib
    Path("models/lstm").mkdir(parents=True, exist_ok=True)
    forecaster.save("models/lstm/")

    # Update meta with input_size
    meta_path = "models/lstm/lstm_meta.pkl"
    meta = joblib.load(meta_path)
    meta["input_size"] = input_size
    joblib.dump(meta, meta_path)

    print("  ✓ LSTM trained and saved")
    return forecaster


def train_lgbm(config: dict, X_train: pd.DataFrame, y_train: pd.Series,
               X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """Train LightGBM quantile regression."""
    print("\n" + "="*50)
    print("STEP 3: LightGBM Quantile Regression")
    print("="*50)

    from src.models.lgbm_quantile import LGBMQuantileForecaster

    lgbm_config = config["lgbm_quantile"]

    print(f"  Quantiles: {lgbm_config['quantiles']}")
    print(f"  Features: {X_train.shape[1]}")

    forecaster = LGBMQuantileForecaster(
        quantiles=lgbm_config["quantiles"],
        n_estimators=lgbm_config["n_estimators"],
        learning_rate=lgbm_config["learning_rate"],
        num_leaves=lgbm_config["num_leaves"],
        min_child_samples=lgbm_config["min_child_samples"],
        subsample=lgbm_config["subsample"],
        colsample_bytree=lgbm_config["colsample_bytree"],
    )

    forecaster.fit(X_train, y_train, X_val, y_val)

    # Feature importance
    importance = forecaster.feature_importance(top_n=10)
    print(f"\n  Top 10 features:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:<25} {row['importance']:>6.0f}")

    Path("models/lgbm").mkdir(parents=True, exist_ok=True)
    forecaster.save("models/lgbm/")
    print("  ✓ LightGBM trained and saved")
    return forecaster


def calibrate_conformal(
    config: dict,
    lgbm_model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Calibrate conformal prediction on validation set."""
    print("\n" + "="*50)
    print("STEP 4: Conformal Calibration (EnbPI)")
    print("="*50)

    import joblib
    from src.evaluation.conformal import EnbPI, SplitConformal

    # Get point predictions from q50
    preds_dict = lgbm_model.predict(X_val)
    y_pred_val = preds_dict[0.50]
    y_val_arr = y_val.values

    conf_config = config["conformal"]

    # EnbPI (primary — handles temporal dependence)
    enbpi = EnbPI(
        alpha=conf_config["alpha"],
        window_size=conf_config["rolling_window"],
    )
    enbpi.initialize(y_val_arr, y_pred_val)

    # Also calibrate 80% intervals
    enbpi_80 = EnbPI(
        alpha=conf_config["alpha_80"],
        window_size=conf_config["rolling_window"],
    )
    enbpi_80.initialize(y_val_arr, y_pred_val)

    # Split conformal (for comparison)
    split_conf = SplitConformal(alpha=conf_config["alpha"])
    split_conf.calibrate(y_val_arr, y_pred_val)

    Path("models/conformal").mkdir(parents=True, exist_ok=True)
    joblib.dump(enbpi, "models/conformal/enbpi_95.pkl")
    joblib.dump(enbpi_80, "models/conformal/enbpi_80.pkl")
    joblib.dump(split_conf, "models/conformal/split_conformal.pkl")

    print(f"  ✓ EnbPI calibrated (q̂={enbpi.q_hat:.1f})")
    print(f"  ✓ EnbPI 80% calibrated (q̂={enbpi_80.q_hat:.1f})")
    print(f"  ✓ Split conformal calibrated (q̂={split_conf.q_hat:.1f})")


def main(config_path: str):
    print("PROBABILISTIC FORECASTING ENGINE — Training Pipeline")
    print("="*60)

    config = load_config(config_path)
    df = load_data(config)

    # Feature engineering
    print("\nBuilding feature matrix...")
    from src.utils.features import build_feature_matrix, get_feature_columns, train_val_test_split

    df_features = build_feature_matrix(
        df,
        target_col=config["data"]["target_col"],
        date_col=config["data"]["date_col"],
        lag_hours=config["features"]["lag_hours"],
        rolling_windows=config["features"]["rolling_windows"],
    )

    print(f"Feature matrix: {df_features.shape[0]:,} rows × {df_features.shape[1]} cols")

    # Train/val/test split
    train, val, test = train_val_test_split(
        df_features,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
    )

    feature_cols = get_feature_columns(df_features, config["data"]["target_col"])
    target_col = config["data"]["target_col"]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    # Save test set for evaluation
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    test.to_csv("data/processed/test_set.csv", index=False)

    # Train models
    np_model = train_neuralprophet(df, config)
    lstm_model = train_lstm(
        df, config,
        X_train.values, y_train.values,
        X_val.values, y_val.values,
    )
    lgbm_model = train_lgbm(config, X_train, y_train, X_val, y_val)
    calibrate_conformal(config, lgbm_model, X_val, y_val)

    print("\n" + "="*60)
    print("✓ ALL MODELS TRAINED")
    print("  Next: python scripts/evaluate.py")
    print("  Then: uvicorn src.serving.api:app --reload --port 8000")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
