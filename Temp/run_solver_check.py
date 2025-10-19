import yaml
from pathlib import Path
from mfg_finance.grid import Grid1D
from mfg_finance.models.hft import HFTParams, eta_from_m_alpha, initial_density
from mfg_finance.solver import solve_mfg_picard

root = Path('mfg-finance')
config_path = root / 'configs' / 'baseline.yaml'
with config_path.open('r', encoding='utf-8') as fp:
    cfg = yaml.safe_load(fp)

grid_cfg = cfg['grid']
params_cfg = cfg['params']
solver_cfg = cfg.get('solver', {})

grid = Grid1D(
    x_min=float(grid_cfg['x_min']),
    x_max=float(grid_cfg['x_max']),
    nx=int(grid_cfg['nx']),
    T=float(grid_cfg['T']),
    nt=int(grid_cfg['nt']),
    bc=str(grid_cfg.get('bc', 'neumann')),
)

params = HFTParams(
    nu=float(params_cfg['nu']),
    phi=float(params_cfg['phi']),
    gamma_T=float(params_cfg['gamma_T']),
    eta0=float(params_cfg.get('eta0', 0.5)),
    eta1=float(params_cfg.get('eta1', 0.8)),
    m0_mean=float(params_cfg.get('m0_mean', 0.0)),
    m0_std=float(params_cfg.get('m0_std', 1.0)),
)

m0 = initial_density(grid, params)
U_all, M_all, alpha_all, errors, metrics = solve_mfg_picard(
    grid,
    params,
    max_iter=int(solver_cfg.get('max_iter', 200)),
    tol=float(solver_cfg.get('tol', 1e-7)),
    mix=0.6,
    m0=m0,
    eta_callback=eta_from_m_alpha,
)

print('iterations:', len(errors))
print('final_error:', errors[-1] if errors else None)
print('mean_abs_alpha:', metrics['mean_abs_alpha'])
