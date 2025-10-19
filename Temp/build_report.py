import json
from pathlib import Path

REPORT_DIRS = sorted(Path('notebooks_output').glob('run-*'))
if not REPORT_DIRS:
    raise SystemExit('No notebooks_output/run-* directories found.')
REPORT_DIR = REPORT_DIRS[-1]
print('Using report directory:', REPORT_DIR)

metrics_path = REPORT_DIR / 'metrics.json'
calibration_path = REPORT_DIR / 'calibration.json'
if not metrics_path.exists() or not calibration_path.exists():
    raise SystemExit('Missing metrics.json or calibration.json in report directory.')

metrics = json.loads(metrics_path.read_text())
calib = json.loads(calibration_path.read_text())

plots_dir = Path('reports/analytics')
annual_vol_hist = (plots_dir / 'annual_vol_hist.png').resolve()
top20_volatility = (plots_dir / 'top20_volatility.png').resolve()
mean_return_hist = (plots_dir / 'mean_return_hist.png').resolve()
median_roll_vol_time = (plots_dir / 'median_roll_vol_time.png').resolve()

density = (REPORT_DIR / 'density.png').resolve()
value = (REPORT_DIR / 'value.png').resolve()
alpha_cuts = (REPORT_DIR / 'alpha_cuts.png').resolve()
convergence = (REPORT_DIR / 'convergence.png').resolve()
price_block = ''
price_fig = REPORT_DIR / 'price.png'
if price_fig.exists():
    price_block = f"\n        <h3>Preço endógeno</h3>\n        <img src=\"{price_fig}\" alt=\"Preço\" />\n        <p>Preço médio {metrics.get('price_mean', 0):.3f} com desvio {metrics.get('price_std', 0):.3f}.</p>\n"

template = Path('Temp/report_template.html').read_text(encoding='utf-8')
html = template.format(
    nu=calib['nu'],
    phi=calib['phi'],
    gamma_T=calib['gamma_T'],
    mix=0.6,
    iterations=metrics['iterations'],
    final_error=metrics['final_error'],
    mean_abs_alpha=metrics['mean_abs_alpha'],
    liquidity_proxy=metrics['liquidity_proxy'],
    annual_vol_hist=annual_vol_hist,
    top20_volatility=top20_volatility,
    mean_return_hist=mean_return_hist,
    median_roll_vol_time=median_roll_vol_time,
    density=density,
    value=value,
    alpha_cuts=alpha_cuts,
    convergence=convergence,
    price_block=price_block,
    vol_median=calib['median_annual_vol'],
    timestamp=REPORT_DIR.name,
)

output_html = REPORT_DIR / 'summary.html'
output_html.write_text(html, encoding='utf-8')
print('Report saved to', output_html)
