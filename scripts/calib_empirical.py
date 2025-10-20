import json
from pathlib import Path

import numpy as np
import pandas as pd

cotahist_dir = Path('data/b3')
files = sorted(cotahist_dir.glob('COTAHIST_A*.TXT'))
if not files:
    raise SystemExit('No COTAHIST files in data/b3')

records = []
for fp in files:
    with fp.open('r', encoding='latin-1') as fh:
        for line in fh:
            if not line.startswith('01'):
                continue
            date = pd.to_datetime(line[2:10], format='%Y%m%d', errors='coerce')
            if date is pd.NaT:
                continue
            bdi = line[10:12]
            if bdi not in {'02', '03'}:
                continue
            ticker = line[12:24].strip()
            if not ticker:
                continue
            open_px = int(line[56:69]) / 10000
            high_px = int(line[69:82]) / 10000
            low_px = int(line[82:95]) / 10000
            avg_px = int(line[95:108]) / 10000
            close_px = int(line[108:121]) / 10000
            best_bid = int(line[121:134]) / 10000
            best_ask = int(line[134:147]) / 10000
            trades = int(line[147:152])
            volume_shares = int(line[152:170])
            volume_money = int(line[170:188]) / 100
            records.append((date, ticker, open_px, high_px, low_px, avg_px, close_px,
                            best_bid, best_ask, trades, volume_shares, volume_money))

if not records:
    raise SystemExit('No equity records parsed.')

cols = ['date','asset','open','high','low','avg','close','best_bid','best_ask','trades','volume_shares','volume_money']
raw_df = pd.DataFrame(records, columns=cols)
raw_df.sort_values(['asset','date'], inplace=True)
raw_df.reset_index(drop=True, inplace=True)

processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)
raw_df.to_parquet(processed_dir / 'cotahist_equities_extended.parquet', index=False)

# Spread proxy
best_spread = (raw_df['best_ask'] - raw_df['best_bid']).clip(lower=0)
safe_close = raw_df['close'].clip(lower=1e-4)
range_spread = (raw_df['high'] - raw_df['low']).clip(lower=0)
raw_df['spread_rel'] = np.where(best_spread > 1e-4, best_spread / safe_close, range_spread / safe_close)
raw_df['spread_rel'] = raw_df['spread_rel'].clip(lower=1e-6)
raw_df['dollar_volume'] = raw_df['volume_money']

agg = raw_df.groupby('asset').agg(
    spread_rel=('spread_rel', 'median'),
    volume_shares_mean=('volume_shares', 'mean'),
    dollar_volume_mean=('dollar_volume', 'mean'),
).replace([np.inf, -np.inf], np.nan).dropna().reset_index()

spread_q25 = float(agg['spread_rel'].quantile(0.25))
spread_q50 = float(agg['spread_rel'].quantile(0.50))
spread_q75 = float(agg['spread_rel'].quantile(0.75))
spread_q90 = float(agg['spread_rel'].quantile(0.90))

eta0 = max(spread_q25, 1e-4)
eta1 = max(spread_q75 - spread_q25, 1e-4)
phi = max(spread_q50 ** 2, 1e-5)
# use half-spread squared for nu (diffusion proxy)
nu = max((spread_q50 / 2) ** 2, 1e-6)
gamma_T = max(spread_q90, spread_q50)

calibration = {
    'nu': round(nu, 6),
    'phi': round(phi, 6),
    'eta0': round(eta0, 6),
    'eta1': round(eta1, 6),
    'gamma_T': round(gamma_T, 6),
    'spread_quantiles': {
        '25pct': spread_q25,
        '50pct': spread_q50,
        '75pct': spread_q75,
        '90pct': spread_q90,
    },
    'avg_volume_shares': float(agg['volume_shares_mean'].mean()),
    'avg_dollar_volume': float(agg['dollar_volume_mean'].mean()),
}

analytics_dir = Path('reports/analytics')
analytics_dir.mkdir(parents=True, exist_ok=True)
(analytics_dir / 'calibration_empirical.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
rows = []
for q in quantiles:
    rows.append({
        'quantile': q,
        'volume_shares': float(agg['volume_shares_mean'].quantile(q)),
        'dollar_volume': float(agg['dollar_volume_mean'].quantile(q)),
        'spread_rel': float(agg['spread_rel'].quantile(q)),
    })
pd.DataFrame(rows).to_csv(processed_dir / 'supply_curve.csv', index=False)

# update baseline config
def update_baseline():
    import yaml
    cfg_path = Path('mfg-finance/configs/baseline.yaml')
    with cfg_path.open('r', encoding='utf-8') as fp:
        cfg = yaml.safe_load(fp)
    params = cfg.setdefault('params', {})
    params['nu'] = calibration['nu']
    params['phi'] = calibration['phi']
    params['eta0'] = calibration['eta0']
    params['eta1'] = calibration['eta1']
    params['gamma_T'] = calibration['gamma_T']
    with cfg_path.open('w', encoding='utf-8') as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False)

update_baseline()

print('Empirical calibration:', calibration)
print('Supply curve saved to data/processed/supply_curve.csv')
