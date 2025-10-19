import json
from pathlib import Path
import pandas as pd

cotahist_dir = Path('data/b3')
files = sorted(cotahist_dir.glob('COTAHIST_A*.TXT'))
if not files:
    raise SystemExit(f'No COTAHIST files found in {cotahist_dir}')

records = []
for file_path in files:
    with file_path.open('r', encoding='latin-1') as fh:
        for line in fh:
            if not line.startswith('01'):
                continue
            date = pd.to_datetime(line[2:10], format='%Y%m%d', errors='coerce')
            ticker = line[12:24].strip()
            close = int(line[108:121]) / 100.0
            if date is pd.NaT or not ticker:
                continue
            records.append((date, ticker, close))

if not records:
    raise SystemExit('No trading records parsed from COTAHIST input.')

df = pd.DataFrame(records, columns=['date', 'asset', 'close'])

# basic cleaning
initial_len = len(df)
df = df[df['close'] > 0].copy()
df.sort_values(['asset', 'date'], inplace=True)
df.drop_duplicates(subset=['asset', 'date'], inplace=True)

df.reset_index(drop=True, inplace=True)

processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)
parquet_path = processed_dir / 'cotahist_2015_2025_clean.parquet'
df.to_parquet(parquet_path, index=False)

summary = {
    'files_processed': [p.name for p in files],
    'rows_clean': int(len(df)),
    'rows_removed_zero_close': int(initial_len - len(df)),
    'assets': int(df['asset'].nunique()),
    'date_start': df['date'].min().strftime('%Y-%m-%d'),
    'date_end': df['date'].max().strftime('%Y-%m-%d'),
}

summary_path = processed_dir / 'cotahist_summary.json'
summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

print(json.dumps(summary, indent=2))
