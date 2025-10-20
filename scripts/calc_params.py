import pandas as pd
summary = pd.read_csv('data/processed/cotahist_asset_summary.csv')
summary = summary.replace([float('inf'), float('-inf')], pd.NA)
summary = summary.dropna(subset=['std_log_return', 'annualised_vol'])
median_daily_std = float(summary['std_log_return'].median())
median_annual_vol = float(summary['annualised_vol'].median())
upper_quartile_vol = float(summary['annualised_vol'].quantile(0.75))
nu = median_daily_std ** 2
phi = median_annual_vol / 20.0
gamma_T = upper_quartile_vol * 0.5
print({'nu': round(nu, 6), 'phi': round(phi, 6), 'gamma_T': round(gamma_T, 6)})
