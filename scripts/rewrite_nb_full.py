import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

# Cell 2
nb['cells'][2]['source'] = [
    'from __future__ import annotations\n',
    '\n',
    'import json\n',
    'from datetime import datetime\n',
    'from pathlib import Path\n',
    '\n',
    'import matplotlib.pyplot as plt\n',
    'import numpy as np\n',
    'import pandas as pd\n',
    'import yaml\n',
    '\n',
    'from IPython.display import display\n',
    'import pprint\n',
    '\n',
    'from mfg_finance.grid import Grid1D\n',
    'from mfg_finance.models.hft import HFTParams, eta_from_m_alpha, initial_density\n',
    'from mfg_finance.price import solve_price_clearing\n',
    'from mfg_finance.solver import solve_mfg_picard\n',
    'from mfg_finance.viz import plot_alpha_cuts, plot_convergence, plot_density_time, plot_price, plot_value_time\n'
]

# Cell 4
nb['cells'][4]['source'] = [
    "ROOT = Path.cwd()\n",
    "DATA_PROCESSED = ROOT / 'data' / 'processed'\n",
    "OUTPUT_BASE = ROOT / 'notebooks_output'\n",
    "OUTPUT_BASE.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "timestamp = datetime.now().strftime('run-%Y%m%d-%H%M%S')\n",
    "REPORT_DIR = OUTPUT_BASE / timestamp\n",
    "REPORT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "print(f'Resultados serão salvos em: {REPORT_DIR}')\n",
    "REPORT_DIR\n"
]

# Cell 6
nb['cells'][6]['source'] = [
    "asset_summary = pd.read_csv(DATA_PROCESSED / 'cotahist_asset_summary.csv')\n",
    "asset_summary = asset_summary.replace([np.inf, -np.inf], np.nan).dropna(subset=['std_log_return', 'annualised_vol'])\n",
    "median_daily_std = float(asset_summary['std_log_return'].median())\n",
    "median_annual_vol = float(asset_summary['annualised_vol'].median())\n",
    "upper_quartile_vol = float(asset_summary['annualised_vol'].quantile(0.75))\n",
    "\n",
    "nu = median_daily_std ** 2\n",
    "phi = median_annual_vol / 20.0\n",
    "gamma_T = upper_quartile_vol * 0.5\n",
    "\n",
    "calibration_summary = {\n",
    "    'median_daily_std': median_daily_std,\n",
    "    'median_annual_vol': median_annual_vol,\n",
    "    'upper_quartile_vol': upper_quartile_vol,\n",
    "    'nu': round(nu, 6),\n",
    "    'phi': round(phi, 6),\n",
    "    'gamma_T': round(gamma_T, 6),\n",
    "}\n",
    "with open(REPORT_DIR / 'calibration.json', 'w', encoding='utf-8') as fp:\n",
    "    json.dump(calibration_summary, fp, indent=2)\n",
    "print('Calibração heurística:')\n",
    "print(json.dumps(calibration_summary, indent=2))\n",
    "calibration_summary\n"
]

# Cell 8
nb['cells'][8]['source'] = [
    "baseline_path = ROOT / 'mfg-finance' / 'configs' / 'baseline.yaml'\n",
    "with open(baseline_path, 'r', encoding='utf-8') as fp:\n",
    "    baseline_cfg = yaml.safe_load(fp)\n",
    "baseline_cfg.setdefault('params', {})\n",
    "baseline_cfg['params']['nu'] = calibration_summary['nu']\n",
    "baseline_cfg['params']['phi'] = calibration_summary['phi']\n",
    "baseline_cfg['params']['gamma_T'] = calibration_summary['gamma_T']\n",
    "with open(baseline_path, 'w', encoding='utf-8') as fp:\n",
    "    yaml.safe_dump(baseline_cfg, fp, sort_keys=False)\n",
    "print('Parâmetros atualizados do baseline:')\n",
    "print(json.dumps(baseline_cfg['params'], indent=2))\n",
    "baseline_cfg['params']\n"
]

# Cell 10
nb['cells'][10]['source'] = [
    "grid_cfg = baseline_cfg['grid']\n",
    "solver_cfg = baseline_cfg.get('solver', {})\n",
    "params_cfg = baseline_cfg['params']\n",
    "grid = Grid1D(\n",
    "    x_min=float(grid_cfg['x_min']),\n",
    "    x_max=float(grid_cfg['x_max']),\n",
    "    nx=int(grid_cfg['nx']),\n",
    "    T=float(grid_cfg['T']),\n",
    "    nt=int(grid_cfg['nt']),\n",
    "    bc=str(grid_cfg.get('bc', 'neumann')),\n",
    ")\n",
    "params = HFTParams(\n",
    "    nu=float(params_cfg['nu']),\n",
    "    phi=float(params_cfg['phi']),\n",
    "    gamma_T=float(params_cfg['gamma_T']),\n",
    "    eta0=float(params_cfg.get('eta0', 0.5)),\n",
    "    eta1=float(params_cfg.get('eta1', 0.8)),\n",
    "    m0_mean=float(params_cfg.get('m0_mean', 0.0)),\n",
    "    m0_std=float(params_cfg.get('m0_std', 1.0)),\n",
    ")\n",
    "print('Grid configurado:')\n",
    "print(grid)\n",
    "print('Parâmetros HFT:')\n",
    "print(params)\n",
    "grid, params\n"
]

# Cell 12
nb['cells'][12]['source'] = [
    "m0 = initial_density(grid, params)\n",
    "hjb_kwargs = {\n",
    "    'max_inner': 3,\n",
    "    'tol': 1e-8,\n",
    "    'max_dissipation': 1.0,\n",
    "    'alpha_cap': 1.0,\n",
    "    'value_cap': 40.0,\n",
    "    'value_relaxation': 0.5,\n",
    "}\n",
    "solver_mix = float(solver_cfg.get('mix', 0.6))\n",
    "U_all, M_all, alpha_all, errors, metrics = solve_mfg_picard(\n",
    "    grid,\n",
    "    params,\n",
    "    max_iter=int(solver_cfg.get('max_iter', 120)),\n",
    "    tol=float(solver_cfg.get('tol', 1e-6)),\n",
    "    mix=solver_mix,\n",
    "    m0=m0,\n",
    "    hjb_kwargs=hjb_kwargs,\n",
    "    eta_callback=eta_from_m_alpha,\n",
    ")\n",
    "run_metrics = {\n",
    "    'iterations': len(errors),\n",
    "    'final_error': float(errors[-1]) if errors else None,\n",
    "    **metrics,\n",
    "}\n",
    "print('Métricas da simulação:')\n",
    "for k, v in run_metrics.items():\n",
    "    print(f'  {k}: {v}')\n",
    "run_metrics\n"
]

# Cell 14
nb['cells'][14]['source'] = [
    "compute_price = True\n",
    "price_results = None\n",
    "if compute_price:\n",
    "    supply_schedule = np.zeros(len(grid.t))\n",
    "    sensitivity = 0.2\n",
    "    def alpha_field(idx: int, price: float) -> np.ndarray:\n",
    "        return alpha_all[idx] - sensitivity * price\n",
    "    prices = solve_price_clearing(alpha_field, M_all, supply_schedule, grid.dx)\n",
    "    price_results = prices\n",
    "    run_metrics['price_mean'] = float(np.mean(prices))\n",
    "    run_metrics['price_std'] = float(np.std(prices))\n",
    "if price_results is not None:\n",
    "    print(f\"Preço médio: {run_metrics.get('price_mean', 0.0):.4f} | desvio: {run_metrics.get('price_std', 0.0):.4f}\")\n",
    "else:\n",
    "    print('Clearing de preço desativado.')\n",
    "run_metrics\n"
]

# Cell 16
nb['cells'][16]['source'] = [
    "np.save(REPORT_DIR / 'U_all.npy', U_all)\n",
    "np.save(REPORT_DIR / 'M_all.npy', M_all)\n",
    "np.save(REPORT_DIR / 'alpha_all.npy', alpha_all)\n",
    "with open(REPORT_DIR / 'metrics.json', 'w', encoding='utf-8') as fp:\n",
    "    json.dump(run_metrics, fp, indent=2)\n",
    "if price_results is not None:\n",
    "    np.savetxt(REPORT_DIR / 'price.csv', np.column_stack((grid.t, price_results)), delimiter=',', header='time,price', comments='')\n",
    "print('Arquivos gerados:')\n",
    "for name in sorted(p.name for p in REPORT_DIR.iterdir()):\n",
    "    print('  -', name)\n",
    "sorted(p.name for p in REPORT_DIR.iterdir())\n"
]

# Cell 18
nb['cells'][18]['source'] = [
    "plot_density_time(M_all, grid, REPORT_DIR / 'density.png')\n",
    "plot_value_time(U_all, grid, REPORT_DIR / 'value.png')\n",
    "plot_alpha_cuts(alpha_all, grid, times=[0.0, 0.25 * grid.T, 0.5 * grid.T], path=REPORT_DIR / 'alpha_cuts.png')\n",
    "plot_convergence(errors, REPORT_DIR / 'convergence.png')\n",
    "if price_results is not None:\n",
    "    plot_price(grid.t, price_results, REPORT_DIR / 'price.png')\n",
    "print('Figuras atualizadas em', REPORT_DIR)\n",
    "sorted(p.name for p in REPORT_DIR.iterdir())\n"
]

# Cell 20
nb['cells'][20]['source'] = [
    "summary_display = {\n",
    "    'output_dir': str(REPORT_DIR),\n",
    "    'iterations': run_metrics.get('iterations'),\n",
    "    'final_error': run_metrics.get('final_error'),\n",
    "    'mean_abs_alpha': run_metrics.get('mean_abs_alpha'),\n",
    "    'std_alpha': run_metrics.get('std_alpha'),\n",
    "    'liquidity_proxy': run_metrics.get('liquidity_proxy'),\n",
    "    'price_mean': run_metrics.get('price_mean'),\n",
    "    'price_std': run_metrics.get('price_std'),\n",
    "}\n",
    "print('Resumo final:')\n",
    "for k, v in summary_display.items():\n",
    "    print(f'  {k}: {v}')\n",
    "summary_display\n"
]

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
