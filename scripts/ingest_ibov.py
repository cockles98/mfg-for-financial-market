import json
from pathlib import Path
import pandas as pd

csv_path = Path('data/ibov_b3_2017_2025.csv')
if not csv_path.exists():
    raise SystemExit(f'CSV not found: {csv_path}')

raw_df = pd.read_csv(csv_path)
raw_len = len(raw_df)
raw_df.columns = [c.strip().lower() for c in raw_df.columns]
expected = {'date', 'asset', 'close'}
missing = expected - set(raw_df.columns)
if missing:
    raise SystemExit(f'Missing required columns: {missing}')

raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
raw_df['asset'] = raw_df['asset'].astype(str).str.strip().str.upper()
raw_df['close'] = pd.to_numeric(raw_df['close'], errors='coerce')

clean_df = raw_df.dropna(subset=['date', 'asset', 'close'])
clean_df = clean_df.sort_values(['asset', 'date']).drop_duplicates(subset=['asset', 'date'])
clean_df = clean_df.reset_index(drop=True)

processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)
parquet_path = processed_dir / 'ibov_b3_2017_2025_clean.parquet'
clean_df.to_parquet(parquet_path, index=False)

summary = {
    'rows_original': int(raw_len),
    'rows_clean': int(len(clean_df)),
    'duplicates_removed': int(raw_len - len(clean_df)),
    'assets': int(clean_df['asset'].nunique()),
    'date_start': clean_df['date'].min().strftime('%Y-%m-%d'),
    'date_end': clean_df['date'].max().strftime('%Y-%m-%d'),
    'missing_close_zero': int((clean_df['close'] == 0).sum()),
}

summary_path = processed_dir / 'ibov_summary.json'
summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

print(json.dumps(summary, indent=2))
