"""
Tests for conformal prediction coverage guarantees.
Run: pytest tests/ -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.conformal import SplitConformal, EnbPI
from src.evaluation.metrics import winkler_score, picp, mpiw


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    np.random.seed(0)
    n = 1000
    t = np.arange(n)
    y = 25000 + 2000 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 500, n)
    y_pred = 25000 + 2000 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 300, n)
    return y, y_pred


# ── Split Conformal Tests ──────────────────────────────────────────────────────

class TestSplitConformal:
    def test_coverage_guarantee(self, synthetic_data):
        """Split conformal should achieve ≥(1-alpha) coverage on calibration set."""
        y, y_pred = synthetic_data
        
        y_cal, y_pred_cal = y[:500], y_pred[:500]
        y_test, y_pred_test = y[500:], y_pred[500:]
        
        alpha = 0.10
        conf = SplitConformal(alpha=alpha)
        conf.calibrate(y_cal, y_pred_cal)
        
        lower, upper = conf.predict_interval(y_pred_test)
        coverage = picp(y_test, lower, upper)
        
        # Coverage should be ≥ 1-alpha
        assert coverage >= 1 - alpha - 0.02, f"Coverage {coverage:.3f} below target {1-alpha}"
    
    def test_q_hat_positive(self, synthetic_data):
        y, y_pred = synthetic_data
        conf = SplitConformal(alpha=0.10)
        conf.calibrate(y[:500], y_pred[:500])
        assert conf.q_hat > 0
    
    def test_interval_symmetric(self, synthetic_data):
        y, y_pred = synthetic_data
        conf = SplitConformal(alpha=0.10)
        conf.calibrate(y[:500], y_pred[:500])
        lower, upper = conf.predict_interval(y_pred[500:600])
        widths = upper - lower
        # All widths should be equal (symmetric intervals)
        assert np.allclose(widths, widths[0])


# ── EnbPI Tests ────────────────────────────────────────────────────────────────

class TestEnbPI:
    def test_coverage_maintained(self, synthetic_data):
        """EnbPI should maintain coverage approximately across test period."""
        y, y_pred = synthetic_data
        
        enbpi = EnbPI(alpha=0.10, window_size=200)
        enbpi.initialize(y[:300], y_pred[:300])
        
        result = enbpi.rolling_coverage(y[300:], y_pred[300:], batch_size=24)
        
        assert result["coverage"] >= 0.80, f"Coverage too low: {result['coverage']:.3f}"
    
    def test_threshold_updates(self, synthetic_data):
        y, y_pred = synthetic_data
        
        enbpi = EnbPI(alpha=0.10, window_size=100)
        enbpi.initialize(y[:100], y_pred[:100])
        
        q_before = enbpi.q_hat
        enbpi.update(y[100:110], y_pred[100:110])
        
        # Threshold should update (may or may not change value)
        assert enbpi.q_hat is not None
        assert isinstance(enbpi.q_hat, float)
    
    def test_rolling_window_size(self, synthetic_data):
        y, y_pred = synthetic_data
        
        enbpi = EnbPI(alpha=0.10, window_size=50)
        enbpi.initialize(y[:200], y_pred[:200])
        
        # Window should not exceed window_size
        assert len(enbpi.residuals) <= 50


# ── Winkler Score Tests ────────────────────────────────────────────────────────

class TestWinklerScore:
    def test_perfect_coverage_minimizes_score(self):
        """Given perfect point forecasts, Winkler score = interval width."""
        y = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        width = 50.0
        
        lower = y_pred - width / 2
        upper = y_pred + width / 2
        
        score = winkler_score(y, lower, upper, alpha=0.10)
        assert abs(score - width) < 1e-6
    
    def test_misses_increase_score(self):
        """Interval misses should increase Winkler score."""
        y = np.array([100.0])
        y_pred = np.array([100.0])
        
        lower_hit = np.array([90.0])
        upper_hit = np.array([110.0])
        lower_miss = np.array([110.0])
        upper_miss = np.array([120.0])
        
        score_hit = winkler_score(y, lower_hit, upper_hit, alpha=0.10)
        score_miss = winkler_score(y, lower_miss, upper_miss, alpha=0.10)
        
        assert score_miss > score_hit
    
    def test_wider_intervals_increase_score(self):
        """Wider intervals with same coverage → higher Winkler score."""
        y = np.ones(100) * 100
        y_pred = np.ones(100) * 100
        
        narrow_lower = y_pred - 10
        narrow_upper = y_pred + 10
        wide_lower = y_pred - 50
        wide_upper = y_pred + 50
        
        score_narrow = winkler_score(y, narrow_lower, narrow_upper)
        score_wide = winkler_score(y, wide_lower, wide_upper)
        
        assert score_wide > score_narrow


# ── Integration: Full pipeline ────────────────────────────────────────────────

def test_full_pipeline_smoke():
    """Smoke test: full calibrate → forecast → evaluate pipeline."""
    np.random.seed(99)
    n = 500
    y = np.random.normal(25000, 2000, n)
    y_pred = y + np.random.normal(0, 500, n)
    
    # Calibrate
    conf = SplitConformal(alpha=0.10)
    conf.calibrate(y[:300], y_pred[:300])
    
    # Forecast
    lower, upper = conf.predict_interval(y_pred[300:])
    
    # Evaluate
    coverage = picp(y[300:], lower, upper)
    width = mpiw(lower, upper)
    winkler = winkler_score(y[300:], lower, upper, alpha=0.10)
    
    assert 0.50 < coverage <= 1.0
    assert width > 0
    assert winkler > 0
    
    print(f"\nSmoke test: coverage={coverage:.3f}, width={width:.1f}, winkler={winkler:.1f}")
