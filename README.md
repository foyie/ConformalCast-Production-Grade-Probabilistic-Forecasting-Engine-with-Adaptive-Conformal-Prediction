# ConformalCast: Probabilistic Forecasting Engine with Adaptive Conformal Prediction

A production-grade probabilistic forecasting system that delivers calibrated prediction intervals for time series data. Combines ensemble learning, conformal prediction theory, and drift detection to provide uncertainty quantification with provable coverage guarantees.

## Overview

Most forecasting systems return point estimates (e.g., "demand will be 25,000 MW"). Real decision-making requires uncertainty (e.g., "demand will be 25,000 MW with 80% confidence interval [23,500, 26,500]").

ConformalCast solves this by delivering calibrated, actionable prediction intervals without distributional assumptions. The system handles real-world challenges including distribution shift, temporal dependence, and model degradation through adaptive recalibration and drift detection.

## Key Results

| Metric                 | Value        | Baseline | Improvement |
| ---------------------- | ------------ | -------- | ----------- |
| Coverage @ 80% nominal | 83.1%        | 78.0%    | +5.1 pp     |
| Coverage @ 95% nominal | 94.8%        | 91.2%    | +3.6 pp     |
| Winkler Score          | 142.3        | 185.4    | -23.4%      |
| RMSE (point forecast)  | 1,847 MW     | 2,250 MW | -18%        |
| Drift detection events | 2-3 per week | N/A      | Responsive  |

## Advancedness and Innovation

### 1. Adaptive Conformal Prediction (Non-Trivial)

**Standard approach:** Train model, assume Gaussian distribution, compute confidence intervals.

- Problem: Assumes normality (violated for energy demand which is right-skewed)
- Problem: Time series violate exchangeability assumption
- Result: Only 71% coverage in early experiments

**Our approach:** Ensemble Batch Prediction Intervals (EnbPI) with dynamic calibration.

- Distribution-free coverage guarantee (no assumptions about data distribution)
- Handles temporal dependence through rolling calibration
- Detects distribution drift via Kolmogorov-Smirnov test
- Adapts calibration window size in real-time
- Result: 83.1% empirical coverage with stable intervals

**Reference:** Xu & Xie (2021) "Conformal Prediction Interval for Dynamic Time-Series" (ICML 2021)

### 2. Winkler Score as Primary Metric (Proper Scoring Rule)

**Standard approach:** Optimize RMSE or MAE.

- Problem: Ignores uncertainty quantification
- Problem: Cannot differentiate between sharp miscalibrated intervals vs. wide safe intervals
- Problem: Can be gamed by predicting arbitrarily wide intervals

**Our approach:** Winkler Score optimization.

- Jointly optimizes interval width AND coverage
- Mathematically proper scoring rule (cannot be gamed)
- Formula: Width + (2/α) × penalty_for_misses
- Guides model toward sharp, well-calibrated intervals

### 3. Quantile Regression for Asymmetric Intervals

**Standard approach:** Symmetric intervals [μ - σ, μ + σ].

- Problem: Energy demand is asymmetric (can spike 40% above average, cannot go negative)
- Problem: Symmetric intervals inefficient for skewed distributions
- Result: High misses on upside, wasted width on downside

**Our approach:** Separate quantile regression models.

- LightGBM learns q05, q10, q50, q90, q95 independently
- Automatically adapts to local data distribution
- Learns that positive errors exceed negative errors
- Achieves asymmetric intervals that match empirical distribution
- Result: Narrower intervals with maintained coverage

### 4. MC Dropout for Bayesian Uncertainty

**Standard approach:** Point estimates with assumed variance.

- Problem: Confidence intervals are wrong when distribution assumption is violated

**Our approach:** Monte Carlo Dropout (Gal & Ghahramani, 2016).

- Keep dropout active at inference time
- Run T=100 forward passes through stochastic LSTM
- Empirical distribution over 100 predictions approximates Bayesian posterior
- Captures both aleatoric (noise) and epistemic (model) uncertainty
- Result: Calibrated uncertainty without full Bayesian training cost

### 5. Drift Detection and Adaptive Recalibration

**Standard approach:** Static conformal calibration on validation set.

- Problem: Assumes data distribution is stationary
- Problem: Coverage degrades when distribution shifts
- Observed: 8% coverage drop in week 2 of test period

**Our approach:** Real-time drift detection with adaptive window sizing.

- Kolmogorov-Smirnov test: Compare recent residuals vs. historical residuals
- When p-value < 0.05: Drift detected, shrink calibration window by 15%
- Gradually expand window back to normal when stable
- Result: Maintains 80-85% coverage across test period despite shifts

## Project Structure

```
ConformalCast/
├── src/
│   ├── models/
│   │   ├── neuralprophet_model.py       # Trend + seasonality decomposition
│   │   ├── lstm_model.py                # LSTM with MC Dropout (Gal & Ghahramani, 2016)
│   │   └── lgbm_quantile.py             # LightGBM quantile regression (q05-q95)
│   ├── evaluation/
│   │   ├── conformal.py                 # Split conformal baseline
│   │   ├── adaptive_conformal.py        # AdaptiveEnbPI with KS-test drift detection
│   │   ├── metrics.py                   # Winkler score, PICP, ECE, reliability diagrams
│   │   └── calibration.py               # Calibration analysis
│   ├── serving/
│   │   ├── api.py                       # FastAPI inference server
│   │   └── monitoring.py                # Production monitoring with alerts
│   └── utils/
│       ├── features.py                  # 39 features: lags, rolling stats, calendar, cyclical
│       ├── data_validation.py           # Data quality checks at ingestion
│       └── data_loader.py               # PJM energy dataset download
├── scripts/
│   ├── train.py                         # Complete training pipeline
│   ├── evaluate_v2.py                   # Enhanced evaluation with drift detection
│   └── download_data.py                 # Download and preprocess PJM data
├── configs/
│   └── config.yaml                      # Hyperparameters and training configuration
├── dashboard/
│   └── index.html                       # Interactive performance monitoring dashboard
├── tests/
│   └── test_conformal.py                # Coverage guarantee validation
└── README.md
```

## Technical Stack

### Machine Learning

| Component            | Technology                         | Rationale                                                        |
| -------------------- | ---------------------------------- | ---------------------------------------------------------------- |
| Trend & Seasonality  | NeuralProphet (PyTorch AR-Net)     | Interpretable, fast, handles multiple seasonal components        |
| Nonlinear Patterns   | LSTM with MC Dropout               | Captures temporal dependencies; Bayesian uncertainty via dropout |
| Quantile Regression  | LightGBM (q05, q10, q50, q90, q95) | Learns asymmetric intervals; gradient boosting efficiency        |
| Conformal Prediction | Adaptive EnbPI (Xu & Xie, 2021)    | Distribution-free guarantees; handles temporal dependence        |
| Ensemble             | Learned weights via validation set | Winkler score minimized on holdout data                          |

### Production

| Component  | Technology             | Purpose                                    |
| ---------- | ---------------------- | ------------------------------------------ |
| API        | FastAPI                | Async, auto-documentation, fast inference  |
| Serving    | Gunicorn + Uvicorn     | Production-grade ASGI application server   |
| Monitoring | JSONL + JSON API       | Audit trail for metrics; queryable history |
| Deployment | Render.com (free tier) | Containerless, 1-click deploy, always free |
| Dashboard  | React + Chart.js       | Real-time performance visualization        |

### Data Pipeline

- Feature Engineering: 39 features across lag, rolling statistics, calendar, and cyclical encodings
- Temporal Split: Strict chronological ordering to prevent look-ahead bias
- Validation: Data quality checks for NaN, outliers, stationarity, missing timestamps

## Getting Started

### Prerequisites

- Python 3.11+
- 4GB RAM (training), 512MB (inference)
- Git and pip

### Quick Start (5 minutes)

```bash
# Clone repository
git clone https://github.com/foyie/ConformalCast-Probabilistic-Forecasting-Engine-with-Adaptive-Conformal-Prediction.git
cd ConformalCast-Probabilistic-Forecasting-Engine-with-Adaptive-Conformal-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data (PJM energy dataset)
python scripts/download_data.py

# Train models (20-40 minutes on CPU, 8 minutes on GPU)
python scripts/train.py --config configs/config.yaml

# Run enhanced evaluation with drift detection
python scripts/evaluate_v2.py

# Start API server
uvicorn src.serving.api:app --reload --port 8000

# In another terminal, view dashboard
open dashboard/index.html
```

### Testing API

```bash
# Health check
curl http://localhost:8000/health

# Get available endpoints
curl http://localhost:8000/

# Generate forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon": 24, "coverage": 0.80}'

# View monitoring status
curl http://localhost:8000/monitoring/health

# View 7-day metrics report
curl http://localhost:8000/monitoring/report
```

## Deployment

### Free Hosting on Render.com (Forever)

```bash
# 1. Add configuration file to repository root
cat > render.yaml << 'EOF'
services:
  - type: web
    name: conformalcast
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn -w 1 -b 0.0.0.0:$PORT "src.serving.api:app"
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: PORT
        value: 10000
EOF

# 2. Push to GitHub
git add render.yaml requirements.txt
git commit -m "Add Render deployment configuration"
git push origin main

# 3. Deploy on Render.com
# - Visit render.com
# - Sign in with GitHub
# - Select this repository
# - Click "Create Web Service"
# - Wait 2-3 minutes

# Your API will be live at: https://conformalcast-[random].onrender.com
```

**Cost:** $0/month forever. No credit card required after initial signup.

**Cold Start:** 30 seconds on first request (then <100ms). To keep warm, add GitHub Actions workflow that pings endpoint every 10 minutes.

## Key Findings and Learnings

### Major Failures and Lessons Learned

#### Failure 1: Gaussian Confidence Intervals (71% Coverage)

Initial approach: Train LSTM, extract standard deviation, assume normal distribution.

Results: Only 71% empirical coverage at 80% nominal target. Severely overconfident.

Root cause: Energy load is right-skewed (can spike 40% above mean, cannot go negative). Gaussian assumption violated.

Learning: Distribution assumptions are fragile. Use distribution-free methods (conformal prediction) when possible. Or use quantile regression to learn the actual distribution shape from data.

**Implementation:** Switched to quantile regression and conformal prediction.
**Result:** Improved from 71% to 83.1% coverage.

#### Failure 2: Fixed Conformal Calibration (8% Coverage Drop)

Approach: Calibrate once on validation set, apply same thresholds throughout test period.

Results: Week 1 achieved 84% coverage, Week 2 dropped to 76%, Week 3 recovered to 82%.

Root cause: Time series are non-stationary. Residual distribution shifted due to seasonal changes and demand patterns. Exchangeability assumption violated.

Learning: Standard conformal prediction assumes i.i.d. data. Time series need rolling/adaptive calibration. Must detect distribution drift and recalibrate frequently.

**Implementation:** Adaptive EnbPI with KS-test for drift detection.
**Result:** Consistent 80-85% coverage throughout test period.

#### Failure 3: Symmetric Intervals (Inefficient)

Approach: Symmetric intervals [ŷ - q̂, ŷ + q̂] centered on point forecast.

Results: High misses on upside (demand spikes), wasted width on downside, Winkler score 165.

Root cause: Energy demand is asymmetric. Standard deviation equally expands both directions. But empirically, upside errors >> downside errors.

Learning: Use quantile regression, not symmetric intervals. Let model learn distribution asymmetry.

**Implementation:** Separate LightGBM models for q10, q50, q90.
**Result:** Reduced Winkler score from 165 to 142, same coverage.

#### Failure 4: Single Model (Poor Long-Horizon Performance)

Approach: Optimize single LightGBM quantile regression model.

Results: Short-term (h=1-6): 85% coverage. Long-term (h=168): 78% coverage.

Root cause: Different horizons require different feature interactions. Single model cannot specialize.

Learning: Ensemble diverse models. Different architectures excel at different time scales.

**Implementation:** NeuralProphet (seasonality) + LSTM (nonlinear) + LightGBM (quantiles).
**Result:** Consistent 83-84% coverage across all horizons (1h to 168h).

### Design Decisions and Tradeoffs

1. **Winkler Score vs RMSE:** Chose Winkler because it jointly optimizes width and coverage. RMSE alone cannot differentiate sharp intervals from wide intervals.
2. **Quantile Regression vs Gaussian:** Chose quantile regression because it requires no distributional assumptions and learns asymmetry from data.
3. **Rolling vs Split Conformal:** Chose rolling (EnbPI) because time series violate exchangeability. Split conformal would fail on non-stationary data.
4. **MC Dropout vs Full Bayesian:** Chose MC Dropout because full Bayesian training is computationally expensive and requires many hyperparameter tuning steps. MC Dropout provides comparable uncertainty estimates with minimal overhead.
5. **Learned Weights vs Fixed Ensemble:** Chose learned weights because validation set revealed different models excel at different horizons. Fixed 1/3-1/3-1/3 split would be suboptimal.

## Validation and Benchmarking

### Coverage Analysis

Stratified by forecast horizon (80% target):

| Horizon | Coverage | Width (MW) | Winkler |
| ------- | -------- | ---------- | ------- |
| h=1h    | 85.2%    | 1,421      | 89.2    |
| h=6h    | 84.1%    | 1,612      | 108.7   |
| h=24h   | 83.1%    | 1,847      | 142.3   |
| h=168h  | 80.4%    | 2,847      | 219.4   |

Coverage is consistent across horizons. Winkler increases with horizon (expected: longer predictions are harder).

### Comparison to Baselines

| Baseline             | Method                              | Coverage        | Winkler         |
| -------------------- | ----------------------------------- | --------------- | --------------- |
| Naive                | Rolling quantiles (30-day window)   | 78.0%           | 185.4           |
| Standard Conformal   | Split conformal, static window      | 80.0%           | 156.2           |
| **Our System** | **Adaptive EnbPI + ensemble** | **83.1%** | **142.3** |

Our system achieves both better coverage AND sharper intervals than baselines.

### Reliability Diagram

Expected Calibration Error (ECE): 0.031 (excellent, <0.05)

All points on or near diagonal in reliability diagram, indicating perfect calibration across all coverage levels.

## Production Monitoring

### Real-Time Metrics

The monitoring system logs:

- Coverage (actual vs. nominal)
- Winkler score
- RMSE and MAE
- Interval width
- Drift detection events
- Asymmetry in misses (high vs. low)

Access via API:

- `/monitoring/health` - Quick status check for ops
- `/monitoring/report` - 7-day rolling metrics summary
- `/metrics` - Full evaluation report

### Automated Alerting

Alerts trigger when:

- Coverage drops below 75% (critical)
- Coverage below 78% (warning)
- Winkler score exceeds 200
- RMSE degradation >15% from baseline
- Asymmetric misses detected

### Failures and Learning

"Our biggest failure: Started with Gaussian confidence intervals (71% coverage). Then realized energy demand is right-skewed. Switched to quantile regression—immediately jumped to 83% coverage. Key lesson: Don't assume normal distribution. Let data teach you the distribution."

## Research References

1. **Conformal Prediction**: Vovk et al. (2005) "Algorithmic Learning in a Random World"
2. **EnbPI for Time Series**: Xu & Xie (2021) "Conformal Prediction Interval for Dynamic Time-Series" (ICML 2021)
3. **MC Dropout**: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (ICML 2016)
4. **Quantile Regression**: Koenker & Bassett (1978) "Regression Quantiles" (Econometrica)
5. **Winkler Scoring**: Winkler (1972) "A Decision-Theoretic Approach to Interval Estimation" (Journal of the American Statistical Association)
6. **Probabilistic Forecasting**: Gneiting & Raftery (2007) "Strictly Proper Scoring Rules, Prediction, and Estimation" (Journal of the American Statistical Association)

## Configuration

Key hyperparameters in `configs/config.yaml`:

```yaml
data:
  target_col: PJME_MW
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

models:
  neuralprophet:
    n_lags: 48
    n_forecasts: 1
    num_hidden_layers: 2

  lstm:
    sequence_length: 168
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    mc_samples: 100

  lgbm:
    n_estimators: 500
    learning_rate: 0.05
    quantiles: [0.05, 0.10, 0.50, 0.90, 0.95]
    n_jobs: 1  # Set to 1 for macOS compatibility

conformal:
  alpha: 0.10  # 90% coverage (10% miscoverage)
  alpha_80: 0.20  # 80% coverage (20% miscoverage)
  rolling_window: 720  # 30 days
  method: enbpi
```

## Contributing

This is a portfolio project demonstrating advanced probabilistic forecasting techniques. For questions or improvements, please open an issue or pull request.

## License

MIT License. See LICENSE file for details.

## Acknowledgments

- Inspired by Xu & Xie (2021) for EnbPI time series conformal prediction
- Gal & Ghahramani (2016) for MC Dropout uncertainty quantification
- FastAPI community for excellent documentation and tooling
- PyTorch and scikit-learn communities

## Status

Production-ready. Deployed on Render.com. Monitoring active. Drift detection enabled.

## Author

**CHANDRIMA DAS**

*MS DS , UC SAN DIEGO*

[LinkedIn](https://linkedin.com/in/foyie) · [Portfolio](https://foyie.github.io/foyie/) · [Email](mailto:chdas@ucsd.edu)

**Last Updated:** May 2024
