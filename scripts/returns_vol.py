import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

processed_dir = Path('data/processed')
source_parquet = processed_dir / 'cotahist_equities_1986_2025.parquet'
if not source_parquet.exists():
    raise SystemExit(f'Processed parquet not found: {source_parquet}')

print('Loading parquet...', flush=True)
df = pd.read_parquet(source_parquet)

# Ensure sorting by asset/date
df = df.sort_values(['asset', 'date']).copy()

# Compute log returns and simple returns per asset
log_close = np.log(df['close'].values)
df['log_close'] = log_close

df['log_return'] = (
    df.groupby('asset')['log_close'].diff().replace([np.inf, -np.inf], np.nan)
)
df['simple_return'] = (
    df.groupby('asset')['close'].pct_change().replace([np.inf, -np.inf], np.nan)
)

# 20-day rolling volatility (daily log returns)
rolling_std = (
    df.groupby('asset')['log_return']
    .rolling(window=20, min_periods=5)
    .std()
    .reset_index(level=0, drop=True)
)
df['roll_vol_20'] = rolling_std * math.sqrt(252)  # annualised

# Persist enriched dataset
returns_parquet = processed_dir / 'cotahist_equities_returns.parquet'
df.to_parquet(returns_parquet, index=False)

# Per-asset summary statistics
sqrt_252 = math.sqrt(252)
summary = df.groupby('asset').agg(
    date_start=('date', 'min'),
    date_end=('date', 'max'),
    observations=('date', 'count'),
    mean_close=('close', 'mean'),
    min_close=('close', 'min'),
    max_close=('close', 'max'),
    mean_log_return=('log_return', 'mean'),
    std_log_return=('log_return', 'std'),
    mean_simple_return=('simple_return', 'mean'),
    std_simple_return=('simple_return', 'std'),
    avg_roll_vol20=('roll_vol_20', 'mean'),
)
summary['annualised_vol'] = summary['std_log_return'] * sqrt_252
summary['date_start'] = summary['date_start'].dt.strftime('%Y-%m-%d')
summary['date_end'] = summary['date_end'].dt.strftime('%Y-%m-%d')

asset_summary_csv = processed_dir / 'cotahist_asset_summary.csv'
summary.reset_index().to_csv(asset_summary_csv, index=False)

# Market-wide summary statistics
annual_vol = summary['annualised_vol'].dropna()
mean_daily = summary['mean_log_return'].dropna()
market_summary = {
    'assets': int(len(summary)),
    'observations_total': int(len(df)),
    'mean_daily_return_median': float(mean_daily.median()) if not mean_daily.empty else None,
    'mean_daily_return_mean': float(mean_daily.mean()) if not mean_daily.empty else None,
    'annualised_vol_median': float(annual_vol.median()) if not annual_vol.empty else None,
    'annualised_vol_mean': float(annual_vol.mean()) if not annual_vol.empty else None,
    'annualised_vol_quantiles': {
        '10pct': float(annual_vol.quantile(0.10)) if not annual_vol.empty else None,
        '25pct': float(annual_vol.quantile(0.25)) if not annual_vol.empty else None,
        '50pct': float(annual_vol.quantile(0.50)) if not annual_vol.empty else None,
        '75pct': float(annual_vol.quantile(0.75)) if not annual_vol.empty else None,
        '90pct': float(annual_vol.quantile(0.90)) if not annual_vol.empty else None,
    },
    'roll_vol20_mean': float(df['roll_vol_20'].mean(skipna=True)) if 'roll_vol_20' in df else None,
}

market_summary_path = processed_dir / 'cotahist_market_summary.json'
market_summary_path.write_text(json.dumps(market_summary, indent=2), encoding='utf-8')

print(json.dumps(market_summary, indent=2))
