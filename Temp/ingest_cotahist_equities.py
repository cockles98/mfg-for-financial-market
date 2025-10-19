import json
from pathlib import Path
import pandas as pd

cotahist_dir = Path('data/b3')
files = sorted(cotahist_dir.glob('COTAHIST_A*.TXT'))
if not files:
    raise SystemExit(f'No COTAHIST files found in {cotahist_dir}')

allowed_bdi = {'02', '03'}  # lote padrao e fracionario
records = []
for file_path in files:
    with file_path.open('r', encoding='latin-1') as fh:
        for line in fh:
            if not line.startswith('01'):
                continue
            bdi = line[10:12]
            if bdi not in allowed_bdi:
                continue
            date = pd.to_datetime(line[2:10], format='%Y%m%d', errors='coerce')
            ticker = line[12:24].strip()
            if date is pd.NaT or not ticker:
                continue
            close = int(line[108:121]) / 100.0
            open_price = int(line[56:69]) / 100.0
            high = int(line[69:82]) / 100.0
            low = int(line[82:95]) / 100.0
            avg = int(line[95:108]) / 100.0
            records.append((date, ticker, bdi, open_price, high, low, avg, close))

if not records:
    raise SystemExit('No matching equity records parsed from COTAHIST input.')

df = pd.DataFrame(records, columns=['date', 'asset', 'bdi_code', 'open', 'high', 'low', 'avg_price', 'close'])

initial_len = len(df)
df = df[df['close'] > 0].copy()
df.sort_values(['asset', 'date'], inplace=True)
df.drop_duplicates(subset=['asset', 'date'], inplace=True)
df.reset_index(drop=True, inplace=True)

processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)
parquet_path = processed_dir / 'cotahist_equities_2015_2025.parquet'
df.to_parquet(parquet_path, index=False)

summary = {
    'files_processed': [p.name for p in files],
    'rows_clean': int(len(df)),
    'assets': int(df['asset'].nunique()),
    'date_start': df['date'].min().strftime('%Y-%m-%d'),
    'date_end': df['date'].max().strftime('%Y-%m-%d'),
    'bdi_codes': {code: int((df['bdi_code'] == code).sum()) for code in sorted(df['bdi_code'].unique())},
}

summary_path = processed_dir / 'cotahist_equities_summary.json'
summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

print(json.dumps(summary, indent=2))
