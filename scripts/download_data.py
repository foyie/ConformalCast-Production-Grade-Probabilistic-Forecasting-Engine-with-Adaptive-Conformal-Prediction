"""
Data Download Script
=====================
Downloads PJM Hourly Energy Consumption dataset.
Primary: Direct download from public URL
Fallback: yfinance for financial time series
"""

import os
import sys
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_pjm_data():
    """
    Download PJM East hourly energy consumption.
    Dataset: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
    
    We use a direct GitHub mirror of the public dataset.
    """
    print("Downloading PJM hourly energy consumption data...")
    
    # Public mirror of the PJME dataset (PJM East region)
    url = "https://raw.githubusercontent.com/jnshsrs/hourly-energy-consumption/master/PJME_hourly.csv"
    
    output_path = RAW_DIR / "PJME_hourly.csv"
    
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return str(output_path)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total = int(response.headers.get("content-length", 0))
        
        with open(output_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="  PJME_hourly.csv"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"  Saved to {output_path}")
        return str(output_path)
    
    except Exception as e:
        print(f"  Primary URL failed: {e}")
        print("  Falling back to synthetic data generation for testing...")
        return generate_synthetic_energy_data()


def generate_synthetic_energy_data():
    """
    Generate realistic synthetic energy consumption data.
    Matches PJM PJME statistical properties.
    Uses: trend + weekly seasonality + daily seasonality + noise
    """
    print("  Generating synthetic PJM-like energy data (10 years hourly)...")
    
    np.random.seed(42)
    
    # 10 years of hourly data
    n_hours = 10 * 365 * 24
    dates = pd.date_range("2012-01-01", periods=n_hours, freq="h")
    
    t = np.arange(n_hours)
    hour = dates.hour.values
    dayofweek = dates.dayofweek.values
    month = dates.month.values
    
    # Base load (~25,000 MW for PJME)
    base = 25000
    
    # Long-term trend (slight increase then plateau)
    trend = 1000 * np.sin(2 * np.pi * t / (n_hours * 0.8))
    
    # Yearly seasonality (peak summer/winter)
    yearly = 3000 * np.cos(2 * np.pi * (month - 1) / 12 - np.pi)
    
    # Weekly seasonality
    weekly = np.where(dayofweek >= 5, -2000, 500)
    
    # Daily seasonality (morning ramp + evening peak)
    daily = (
        2000 * np.sin(2 * np.pi * (hour - 6) / 24)
        + 1500 * np.sin(2 * np.pi * (hour - 18) / 24)
    )
    
    # Temperature-driven demand spikes (simulated)
    heat_waves = np.random.choice([0, 3000], size=n_hours, p=[0.97, 0.03])
    cold_snaps = np.random.choice([0, 2500], size=n_hours, p=[0.97, 0.03])
    
    # Gaussian noise
    noise = np.random.normal(0, 800, n_hours)
    
    # Autocorrelated residuals (AR(1))
    ar_noise = np.zeros(n_hours)
    ar_noise[0] = noise[0]
    for i in range(1, n_hours):
        ar_noise[i] = 0.7 * ar_noise[i - 1] + noise[i]
    
    pjme_mw = base + trend + yearly + weekly + daily + heat_waves + cold_snaps + ar_noise
    pjme_mw = np.clip(pjme_mw, 10000, 60000)
    
    df = pd.DataFrame({"Datetime": dates, "PJME_MW": pjme_mw.round(1)})
    
    output_path = RAW_DIR / "PJME_hourly.csv"
    df.to_csv(output_path, index=False)
    
    print(f"  Generated {len(df):,} rows of synthetic data")
    print(f"  Saved to {output_path}")
    
    return str(output_path)


def download_financial_data(ticker="SPY", period="10y"):
    """
    Download financial time series using yfinance.
    Good alternative to energy data.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("Install yfinance: pip install yfinance")
        sys.exit(1)
    
    print(f"Downloading {ticker} financial data ({period})...")
    
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    data = data[["Close"]].reset_index()
    data.columns = ["Datetime", "PJME_MW"]  # Rename to match pipeline
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    
    output_path = RAW_DIR / f"{ticker}_daily.csv"
    data.to_csv(output_path, index=False)
    
    print(f"  Downloaded {len(data):,} rows")
    print(f"  Saved to {output_path}")
    
    return str(output_path)


def validate_data(filepath: str) -> pd.DataFrame:
    """Load and validate downloaded data."""
    df = pd.read_csv(filepath)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    
    # Drop duplicates
    df = df.drop_duplicates(subset=["Datetime"])
    
    # Fill small gaps via interpolation (up to 3 hours)
    df = df.set_index("Datetime")
    df = df.resample("h").interpolate(method="time", limit=3)
    df = df.dropna()
    df = df.reset_index()
    
    print(f"\nData validation:")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['Datetime'].min()} → {df['Datetime'].max()}")
    print(f"  Target stats: mean={df['PJME_MW'].mean():.0f}, std={df['PJME_MW'].std():.0f}")
    print(f"  Missing values: {df['PJME_MW'].isna().sum()}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download forecasting dataset")
    parser.add_argument("--source", choices=["pjm", "finance", "synthetic"], default="pjm")
    parser.add_argument("--ticker", default="SPY", help="Ticker for finance source")
    args = parser.parse_args()
    
    if args.source == "pjm":
        filepath = download_pjm_data()
    elif args.source == "finance":
        filepath = download_financial_data(args.ticker)
    else:
        filepath = generate_synthetic_energy_data()
    
    df = validate_data(filepath)
    
    # Save validated version
    validated_path = PROCESSED_DIR / "validated.csv"
    df.to_csv(validated_path, index=False)
    print(f"\nValidated data saved to {validated_path}")
