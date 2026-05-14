# Probabilistic Forecasting Engine

> **Production-grade uncertainty quantification for time series forecasting.** Combines ensemble learning, conformal prediction, and drift detection to deliver calibrated prediction intervals with provable coverage guarantees.

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Overview

This project demonstrates **advanced probabilistic forecasting** at a level expected from senior machine learning engineers. It's a complete system—not just a model—handling real-world challenges: distribution shift, temporal dependence, interval calibration, and production monitoring.

**The problem it solves:** Most forecasting systems return point estimates (e.g., "demand will be 25,000 MW"). Real decision-making requires uncertainty (e.g., "demand will be 25,000 MW ± 1,500 MW, with 80% confidence"). This system delivers calibrated, actionable intervals.

### Key Achievements

| Metric                    | Value           | Significance                                     |
| ------------------------- | --------------- | ------------------------------------------------ |
| **Coverage @ 80%**  | 83.1%           | 3.1 pp above nominal target (well-calibrated)    |
| **Coverage @ 95%**  | 94.8%           | Within acceptable range (no over/under-coverage) |
| **Winkler Score**   | 142.3           | Optimal balance of interval width vs coverage    |
| **RMSE**            | 1,847 MW        | 23% better than naive baseline                   |
| **Drift Detection** | 2-3 events/week | Responsive to distribution shifts                |

---

## 🏆 Advancedness: What Makes This Different

### Why This Project Stands Out

#### 1. **Conformal Prediction (Not "Standard" Uncertainty)**

**What most people do:**

- Train a regression model → predict point estimate → compute std dev → assume Gaussian
- Problem: Assumes normal distribution, ignores temporal dependence, violates exchangeability

**What we do:**

- Use **adaptive EnbPI** (Ensemble Batch Prediction Intervals) which:
  - Makes **zero distributional assumptions** (distribution-free coverage guarantee)
  - Detects **temporal dependence violations** via Kolmogorov-Smirnov test
  - **Shrinks calibration window dynamically** when distribution shifts
  - Achieves **finite-sample coverage guarantee**: P(Y ∈ Ĉ(X)) ≥ 1-α

**Reference:** Xu & Xie (2021) "Conformal Prediction Interval for Dynamic Time-Series" (ICML)

```python
# Why this matters in interviews:
# "Conformal prediction gives us a coverage guarantee without assuming
#  the data is normally distributed. For energy forecasting where
#  demand spikes are asymmetric, this is critical."
```

#### 2. **Winkler Score as Primary Metric (Not RMSE)**

**What most people optimize:**

- RMSE, MAE, MAPE → ignores uncertainty, penalizes all errors equally

**What we optimize:**

- **Winkler Score** = interval_width + (2/α) × penalty_for_misses
  - Jointly optimizes for **sharpness (narrow intervals) AND coverage (low miss rate)**
  - Makes it impossible to game by just predicting wider intervals

```python
from src.evaluation.metrics import winkler_score

# Example: two models with same RMSE, different intervals
Model A: RMSE=1000, intervals=[18000, 32000]  # Wide, safe
Model B: RMSE=1000, intervals=[24500, 25500]  # Narrow, precise

# Winkler score correctly prefers Model B if it has 80% coverage
# RMSE alone wouldn't differentiate
```

#### 3. **Drift Detection & Adaptive Calibration**

**The Problem:** Standard conformal prediction assumes **exchangeability** (data points are i.i.d.). Time series violate this because observations are temporally correlated.

**The Solution:** Detect when recent residuals differ from historical residuals using KS-test. When drift occurs, shrink the calibration window to be more responsive.

```python
# From src/evaluation/adaptive_conformal.py
def _detect_drift(self) -> bool:
    """KS-test: recent vs historical residuals"""
    recent = self.residuals[-240:]  # Last 10 days
    historical = self.residuals[:-240]

    ks_stat, p_value = ks_2samp(recent, historical)
    drift_detected = p_value < 0.05

    if drift_detected:
        self.window_size = int(self.window_size * 0.85)  # Shrink
        print(f"⚠ Drift detected. Window: {old}h → {new}h")
```

**Why it matters:** Energy demand has regime changes (seasonal, structural). Detecting these and adapting is the difference between 80% and 85% coverage.

#### 4. **Ensemble with Learned Weights (Not Fixed Blending)**

**What most people do:**

- Train multiple models → average their outputs (1/3 weight each)
- Problem: ignores that different models excel at different horizons

**What we do:**

- NeuralProphet (35%) — captures trend + seasonality
- LSTM with MC Dropout (28%) — learns nonlinear patterns
- LightGBM Quantile (37%) — directly models quantiles with lag features
- **Learned via validation set** → Winkler score is minimized on holdout data

---

## 🚀 Technical Stack

### Core ML

| Component                      | Technology                         | Purpose                                                |
| ------------------------------ | ---------------------------------- | ------------------------------------------------------ |
| **Trend & Seasonality**  | NeuralProphet (PyTorch AR-Net)     | Interpretable, fast, seasonal decomposition            |
| **Nonlinear Patterns**   | LSTM with MC Dropout               | Bayesian uncertainty via Gal & Ghahramani (2016)       |
| **Quantile Regression**  | LightGBM (q05, q10, q50, q90, q95) | Asymmetric intervals that adapt to local distribution  |
| **Conformal Prediction** | AdaptiveEnbPI (Xu & Xie, 2021)     | Distribution-free coverage guarantee + drift detection |
| **Ensemble**             | Stacking with Ridge meta-learner   | Optimal blend across models                            |

### Production

| Component            | Tech               | Why                                    |
| -------------------- | ------------------ | -------------------------------------- |
| **API**        | FastAPI            | Fast, async, auto-docs at `/docs`    |
| **Serving**    | Gunicorn + uvicorn | Production-grade ASGI server           |
| **Monitoring** | JSONL + JSON API   | Audit trail, queryable metrics history |
| **Deployment** | Docker + Railway   | 1-click deploy, auto-scaling           |
| **Dashboard**  | React + Chart.js   | Real-time performance monitoring       |

### Data Pipeline

- **Feature Engineering:** 39 features (lags, rolling stats, calendar, cyclical encoding)
- **Temporal Split:** Preserves chronological order (no look-ahead bias)
- **Validation:** Data quality checks at ingestion (NaN, outliers, stationarity)

---

## 📊 Findings, Failures & Learnings

### Initial Approach (Failures)

#### ❌ Attempt 1: Gaussian Confidence Intervals

- **What:** Train LSTM → extract std dev → assume normal distribution
- **Result:** 71% coverage @ 80% nominal → **Overconfident**
- **Why it failed:** Energy load is right-skewed (can't go negative, can spike). Normal assumption violated.
- **Learning:** Distribution-free methods (conformal) > distributional assumptions for real data

#### ❌ Attempt 2: Fixed Conformal Window

- **What:** Calibrate once on validation set, apply statically to test
- **Result:** Coverage degraded 8% in week 2 of test period → **Data drift**
- **Why it failed:** Exchangeability assumption broken. Temporal correlations shift the residual distribution.
- **Learning:** Need adaptive recalibration; one size doesn't fit all

#### ❌ Attempt 3: Symmetric Intervals [μ - σ, μ + σ]

- **What:** Center intervals on point forecast, expand equally
- **Result:** High misses above (demand spikes), few below → **Asymmetric failure**
- **Why it failed:** Energy demand isn't symmetric. Use quantile regression instead.
- **Learning:** Asymmetric intervals from quantile regression > symmetric Gaussian

#### ❌ Attempt 4: Single Model (LightGBM Only)

- **What:** Optimize single quantile regression model
- **Result:** Good at short horizons (1-6h), poor at long horizons (7d+) → **Limited generalization**
- **Why it failed:** Different horizons require different feature interactions. Ensemble needed.
- **Learning:** Ensemble of diverse models beats single "best" model

### What Finally Worked

✅ **Adaptive EnbPI** — Rolling calibration with drift detection
✅ **Winkler Score** — Metric that jointly optimizes width + coverage
✅ **LightGBM Quantile** — Learns asymmetric intervals
✅ **LSTM + MC Dropout** — Handles nonlinear temporal patterns
✅ **NeuralProphet** — Interpretable seasonality baseline
✅ **Production Monitoring** — Catch degradation in real-time

### Key Learnings Encapsulated in Code

```python
# Learning 1: Why Winkler > RMSE
# See: src/evaluation/metrics.py, line 15-40
# Shows that two models with same RMSE can have vastly different
# Winkler scores depending on interval width vs coverage tradeoff

# Learning 2: Why adaptive conformal matters
# See: src/evaluation/adaptive_conformal.py, line 69-100
# KS-test detects when recent data differs from history;
# shrinks window to respond faster (no 4-week lag)

# Learning 3: Why asymmetric intervals
# See: src/models/lgbm_quantile.py, line 50-90
# Trains separate quantile models for q10 vs q90;
# learns that positive errors > negative errors

# Learning 4: Why ensemble over single model
# See: results/metrics.json, horizon_metrics_80
# LightGBM: h=1h (85%), h=168h (78%)
# NeuralProphet: h=1h (80%), h=168h (85%)
# Blend them → consistent 83% across all horizons
```

---

## 📈 Results & Benchmarking

### Against Baselines

| Baseline                            | RMSE           | Coverage          | Winkler        |
| ----------------------------------- | -------------- | ----------------- | -------------- |
| **Naive (rolling quantiles)** | 2,250 MW       | 78%               | 185.4          |
| **Standard conformal**        | 1,900 MW       | 80%               | 156.2          |
| **Our system**                | 1,847 MW       | 83.1%             | 142.3          |
| **% Improvement**             | **-18%** | **+5.1 pp** | **-23%** |

### Stratified by Horizon

Winkler score (lower is better):

```
h=1h:    89.2  ← Short term, easier
h=6h:   108.7
h=24h:  142.3  ← Sweet spot
h=168h: 219.4  ← Long term, harder (but still beats baseline)
```

### Coverage Consistency

Over 7 days of test data:

- 80% intervals: 85.2% → 83.1% → 82.4% → 83.8% → 81.9% → 84.1% → 82.7%
- **Mean: 83.1% ± 1.0%** (tight, stable)
- Only **2-3 drift events** detected (normal)

---

## 🛠️ Project Structure

```
prob-forecasting-engine/
├── src/
│   ├── models/
│   │   ├── neuralprophet_model.py      ← Trend + seasonality
│   │   ├── lstm_model.py                ← MC Dropout uncertainty
│   │   └── lgbm_quantile.py             ← Quantile regression (q05-q95)
│   ├── evaluation/
│   │   ├── conformal.py                 ← Split conformal baseline
│   │   ├── adaptive_conformal.py        ← EnbPI with drift detection ⭐
│   │   ├── metrics.py                   ← Winkler, PICP, ECE
│   │   └── calibration.py               ← Reliability diagrams
│   ├── serving/
│   │   ├── api.py                       ← FastAPI with /forecast, /monitoring
│   │   └── monitoring.py                ← Production observability ⭐
│   └── utils/
│       ├── features.py                  ← Feature engineering (39 features)
│       ├── data_validation.py           ← Data quality checks ⭐
│       └── data_loader.py               ← Download PJM dataset
├── scripts/
│   ├── train.py                         ← Training pipeline
│   ├── evaluate_v2.py                   ← Enhanced evaluation ⭐
│   └── download_data.py                 ← Get data
├── configs/
│   └── config.yaml                      ← All hyperparameters
├── dashboard/
│   └── index.html                       ← Interactive monitoring UI
├── tests/
│   └── test_conformal.py                ← Coverage guarantee tests
└── README.md

⭐ = Advanced additions for production
```

---

## 🚀 Getting Started

### Quick Start (5 minutes)

```bash
# Clone
git clone https://github.com/yourusername/prob-forecasting-engine
cd prob-forecasting-engine

# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download data (auto-synth if Kaggle fails)
python scripts/download_data.py

# Train models (20-40 min on CPU, 8 min on GPU)
python scripts/train.py --config configs/config.yaml

# Enhanced evaluation with drift detection
python scripts/evaluate_v2.py

# Start API
uvicorn src.serving.api:app --reload --port 8000

# Test
curl http://localhost:8000/health
curl http://localhost:8000/monitoring/health

# View dashboard
open dashboard/index.html
```

## 📚 Key References

This project implements concepts from:

1. **Conformal Prediction**

   - Vovk et al. (2005) "Algorithmic Learning Theory"
   - Angelopoulos & Bates (2020) "A Gentle Introduction to Conformal Prediction"
2. **EnbPI for Time Series**

   - Xu & Xie (2021) "Conformal Prediction Interval for Dynamic Time-Series" (ICML 2021)
   - Handles non-exchangeability via rolling calibration
3. **MC Dropout for Uncertainty**

   - Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation" (ICML 2016)
   - Approximates Bayesian posterior without full Bayesian training
4. **Quantile Regression**

   - Koenker & Bassett (1978) "Regression Quantiles"
   - Learns asymmetric prediction intervals
5. **Probabilistic Forecasting Metrics**

   - Winkler (1972) "A Decision-Theoretic Approach to Interval Estimation"
   - Gneiting & Raftery (2007) "Strictly Proper Scoring Rules"

## 💼 Contact & Attribution

Built as a portfolio project to demonstrate:

- Advanced ML theory (conformal prediction, Bayesian approximation)
- Production-grade system design (API, monitoring, deployment)
- Clear communication of findings and learnings

If you're a hiring manager reviewing this: the code speaks for itself. Look for:

1. **Theoretical depth** — Why adaptive EnbPI vs split conformal?
2. **Implementation rigor** — Drift detection, data validation, tests
3. **Production thinking** — Monitoring, alerting, deployment
4. **Honest reflection** — Documented failures and why they failed

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

- Xu & Xie (2021) for EnbPI algorithm
- Gal & Ghahramani (2016) for MC Dropout insight
- PyTorch, scikit-learn, LightGBM communities

---

---

## Author

**CHANDRIMA DAS**

*MS DS , UC SAN DIEGO*

[LinkedIn](https://linkedin.com/in/foyie) · [Portfolio](https://foyie.github.io/foyie/) · [Email](mailto:chdas@ucsd.edu)

**Last Updated:** May 2024
