"""
FastAPI Inference Endpoint
============================
POST /forecast → probabilistic forecast with calibrated intervals

Returns:
  - point_forecast: median prediction
  - lower_80 / upper_80: 80% prediction interval
  - lower_95 / upper_95: 95% prediction interval
  - coverage_score: rolling coverage on recent data
  - model_metadata: which models contributed + weights
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.serving.monitoring import ModelMonitor
import os

# Load environment variables
PORT = int(os.getenv("PORT", 8000))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
monitor = ModelMonitor()

app = FastAPI(
    title="Probabilistic Forecasting Engine",
    description="Production probabilistic time series forecasting with calibrated uncertainty intervals",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models (loaded on startup) ──────────────────────────────────────────────
MODELS = {}
CONFORMAL_CALIBRATOR = None


@app.on_event("startup")
async def load_models():
    """Load trained models on startup."""
    global MODELS, CONFORMAL_CALIBRATOR

    model_path = Path("models/")

    if not model_path.exists():
        print("WARNING: models/ directory not found. Run scripts/train.py first.")
        return

    try:
        from src.models.lgbm_quantile import LGBMQuantileForecaster
        MODELS["lgbm"] = LGBMQuantileForecaster.load("models/lgbm/")
        print("✓ LightGBM quantile models loaded")
    except Exception as e:
        print(f"  LightGBM load failed: {e}")

    try:
        import torch
        from src.models.lstm_model import MCDropoutForecaster
        # Input size determined at training time — stored in metadata
        import joblib
        meta = joblib.load("models/lstm/lstm_meta.pkl")
        MODELS["lstm"] = MCDropoutForecaster.load("models/lstm/", input_size=meta.get("input_size", 30))
        print("✓ LSTM MC Dropout model loaded")
    except Exception as e:
        print(f"  LSTM load failed: {e}")

    print(f"Loaded {len(MODELS)} model(s)")


# ── Request / Response Schemas ──────────────────────────────────────────────

class ForecastRequest(BaseModel):
    horizon: int = Field(default=24, ge=1, le=168, description="Forecast horizon in hours (1–168)")
    coverage: float = Field(default=0.80, ge=0.50, le=0.99, description="Desired interval coverage (0.50–0.99)")
    return_samples: bool = Field(default=False, description="Return MC Dropout posterior samples")
    n_samples: int = Field(default=50, ge=10, le=200, description="Number of MC samples (if return_samples=True)")


class IntervalForecast(BaseModel):
    timestamp: str
    point_forecast: float
    lower: float
    upper: float
    coverage: float
    interval_width: float


class ForecastResponse(BaseModel):
    request_id: str
    model_version: str = "1.0.0"
    generated_at: str
    horizon_hours: int
    nominal_coverage: float
    empirical_coverage_recent: Optional[float]
    forecasts: List[IntervalForecast]
    metrics: dict
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime_seconds: float


STARTUP_TIME = time.time()


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if MODELS else "degraded",
        models_loaded=list(MODELS.keys()),
        uptime_seconds=round(time.time() - STARTUP_TIME, 1),
    )



@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    import uuid

    request_id = str(uuid.uuid4())[:8]
    generated_at = datetime.utcnow().isoformat() + "Z"

    if not MODELS:
        return _demo_forecast(request, request_id, generated_at)

    try:
        forecasts = _generate_forecasts(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

    # ── Monitoring hook ─────────────────────────────
    try:
        y_recent = np.array([...])       # actual recent observed values
        y_pred_recent = np.array([f.point_forecast for f in forecasts])
        lower = np.array([f.lower for f in forecasts])
        upper = np.array([f.upper for f in forecasts])

        monitor.log_batch_metrics(
            y_recent,
            y_pred_recent,
            lower,
            upper
        )
    except Exception as monitor_error:
        print(f"Monitoring failed: {monitor_error}")
    # ───────────────────────────────────────────────

    return ForecastResponse(
        request_id=request_id,
        generated_at=generated_at,
        horizon_hours=request.horizon,
        nominal_coverage=request.coverage,
        empirical_coverage_recent=None,
        forecasts=forecasts,
        metrics={
            "model": "ensemble",
            "n_forecasts": len(forecasts),
        },
        metadata={
            "models_used": list(MODELS.keys()),
            "conformal_method": "enbpi",
        },
    )

# async def forecast(request: ForecastRequest):
#     """
#     Generate probabilistic forecast with calibrated prediction intervals.

#     Uses ensemble of available models (LightGBM + LSTM) with conformal
#     calibration to guarantee coverage.
#     """
#     import uuid

#     request_id = str(uuid.uuid4())[:8]
#     generated_at = datetime.utcnow().isoformat() + "Z"

#     # Generate mock forecast if models not loaded (demo mode)
#     if not MODELS:
#         return _demo_forecast(request, request_id, generated_at)

#     # Build feature matrix for forecast horizon
#     try:
#         forecasts = _generate_forecasts(request)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

#     return ForecastResponse(
#         request_id=request_id,
#         generated_at=generated_at,
#         horizon_hours=request.horizon,
#         nominal_coverage=request.coverage,
#         empirical_coverage_recent=None,
#         forecasts=forecasts,
#         metrics={
#             "model": "ensemble",
#             "n_forecasts": len(forecasts),
#         },
#         metadata={
#             "models_used": list(MODELS.keys()),
#             "conformal_method": "enbpi",
#         },
#     )


def _demo_forecast(request: ForecastRequest, request_id: str, generated_at: str) -> ForecastResponse:
    """
    Demo forecast when models aren't loaded.
    Generates realistic-looking synthetic energy load forecast.
    """
    np.random.seed(42)

    now = datetime.utcnow()
    forecasts = []

    base_load = 28000
    alpha = (1 - request.coverage) / 2

    for h in range(1, request.horizon + 1):
        ts = now + timedelta(hours=h)
        hour = ts.hour

        # Realistic daily pattern
        daily_pattern = (
            2000 * np.sin(2 * np.pi * (hour - 6) / 24)
            + 1000 * np.sin(2 * np.pi * (hour - 18) / 24)
        )
        noise = np.random.normal(0, 300)
        point = base_load + daily_pattern + noise

        # Width grows with horizon
        width = 1500 + h * 50 + np.random.normal(0, 100)
        lower = point - width / 2
        upper = point + width / 2

        forecasts.append(IntervalForecast(
            timestamp=ts.isoformat() + "Z",
            point_forecast=round(float(point), 1),
            lower=round(float(lower), 1),
            upper=round(float(upper), 1),
            coverage=request.coverage,
            interval_width=round(float(width), 1),
        ))

    return ForecastResponse(
        request_id=request_id,
        model_version="demo",
        generated_at=generated_at,
        horizon_hours=request.horizon,
        nominal_coverage=request.coverage,
        empirical_coverage_recent=0.831,
        forecasts=forecasts,
        metrics={
            "rmse": 1847.2,
            "mae": 1341.5,
            "winkler_score": 142.3,
            "picp": 0.831,
        },
        metadata={
            "mode": "demo",
            "note": "Run scripts/train.py to load real models",
        },
    )


def _generate_forecasts(request: ForecastRequest) -> List[IntervalForecast]:
    """Real forecast generation with loaded models."""
    # This would use the actual feature pipeline in production
    # Simplified here — the training pipeline writes the actual implementation
    raise NotImplementedError("Load models first via scripts/train.py")


@app.get("/metrics")
async def get_metrics():
    """Return latest evaluation metrics from the most recent evaluation run."""
    metrics_path = Path("results/metrics.json")

    if not metrics_path.exists():
        return {"error": "No metrics found. Run scripts/evaluate.py first."}

    with open(metrics_path) as f:
        return json.load(f)


@app.get("/models")
async def get_model_info():
    """Return info about loaded models."""
    return {
        "models": {
            name: {"status": "loaded", "type": type(model).__name__}
            for name, model in MODELS.items()
        },
        "total_loaded": len(MODELS),
    }

from src.serving.monitoring import ModelMonitor

monitor = ModelMonitor()

@app.post("/forecast")
async def forecast(request: ForecastRequest):
    # ... generate forecast ...

    # Log metrics
    monitor.log_batch_metrics(y_recent, y_pred_recent, lower, upper)

    return response

@app.get("/monitoring/report")
async def monitoring_report():
    """Return 7-day monitoring summary."""
    return monitor.get_report(days=7)

@app.get("/health")
async def health():
    """Check if models are still good."""
    report = monitor.get_report(days=1)

    if report.get("coverage", {}).get("latest", 0) < 0.75:
        return {"status": "degraded", "reason": "Coverage < 75%"}

    return {"status": "healthy"}

if __name__ == "__main__":
    debug_mode = ENVIRONMENT == "development"
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=debug_mode)
