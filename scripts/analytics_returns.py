import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

processed = Path('data/processed')
summary_path = processed / 'cotahist_asset_summary.csv'
returns_path = processed / 'cotahist_equities_returns.parquet'

summary = pd.read_csv(summary_path)
summary['annualised_vol'] = summary['annualised_vol'].fillna(0)
summary['std_log_return'] = summary['std_log_return'].fillna(0)
summary['avg_roll_vol20'] = summary['avg_roll_vol20'].fillna(0)

# Calibration heuristics
median_daily_std = float(summary['std_log_return'].median())
nu = median_daily_std ** 2  # daily variance proxy
phi = float(summary['annualised_vol'].median() / 10.0)
gamma_T = float(summary['annualised_vol'].quantile(0.75))

calibration = {
    'nu': round(nu, 6),
    'phi': round(phi, 6),
    'gamma_T': round(gamma_T, 6),
    'median_daily_std': median_daily_std,
    'median_annual_vol': float(summary['annualised_vol'].median()),
    'quantiles_annual_vol': {
        'p25': float(summary['annualised_vol'].quantile(0.25)),
        'p50': float(summary['annualised_vol'].quantile(0.50)),
        'p75': float(summary['annualised_vol'].quantile(0.75)),
    },
}

(processed / 'calibration_summary.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')

# Plots directory
plots_dir = Path('reports/analytics')
plots_dir.mkdir(parents=True, exist_ok=True)

# Histogram of annualised volatility (clip to 0-3)
plt.figure(figsize=(8, 4))
clipped = summary['annualised_vol'].clip(upper=3.0)
plt.hist(clipped, bins=60, color='steelblue', edgecolor='black', alpha=0.8)
plt.title('Distribuicao de Volatilidade Anualizada (clipe em 3)')
plt.xlabel('Volatilidade anualizada')
plt.ylabel('Numero de ativos')
plt.tight_layout()
plt.savefig(plots_dir / 'annual_vol_hist.png', dpi=150)
plt.close()

# Top 20 volatilities
plt.figure(figsize=(9, 5))
top20 = summary.nlargest(20, 'annualised_vol')[['asset', 'annualised_vol']]
plt.bar(top20['asset'], top20['annualised_vol'], color='darkorange')
plt.xticks(rotation=90)
plt.ylabel('Volatilidade anualizada')
plt.title('Top 20 ativos mais volateis (2015-2025)')
plt.tight_layout()
plt.savefig(plots_dir / 'top20_volatility.png', dpi=150)
plt.close()

# Mean simple return distribution
plt.figure(figsize=(8, 4))
mu = summary['mean_simple_return'].clip(-0.1, 0.1)
plt.hist(mu, bins=80, color='seagreen', edgecolor='black', alpha=0.8)
plt.title('Distribuicao de retornos simples medios (clipe +/-10% ao dia)')
plt.xlabel('Retorno medio diario')
plt.ylabel('Numero de ativos')
plt.tight_layout()
plt.savefig(plots_dir / 'mean_return_hist.png', dpi=150)
plt.close()

# Time series of median rolling volatility
print('Loading returns parquet for timeseries...', flush=True)
returns_df = pd.read_parquet(returns_path, columns=['date', 'roll_vol_20'])
returns_df = returns_df.dropna(subset=['roll_vol_20'])
median_ts = returns_df.groupby('date')['roll_vol_20'].median().reset_index()
median_ts['year'] = median_ts['date'].dt.year

plt.figure(figsize=(10, 4))
plt.plot(median_ts['date'], median_ts['roll_vol_20'], color='purple', linewidth=1.2)
plt.title('Mediana diaria da volatilidade movel (20 dias)')
plt.xlabel('Data')
plt.ylabel('Volatilidade anualizada (mediana)')
plt.tight_layout()
plt.savefig(plots_dir / 'median_roll_vol_time.png', dpi=150)
plt.close()

# Save aggregated timeseries CSV
median_ts.to_csv(plots_dir / 'median_roll_vol_time.csv', index=False)

print(json.dumps(calibration, indent=2))
