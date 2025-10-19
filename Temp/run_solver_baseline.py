from pathlib import Path
from mfg_finance.grid import Grid1D
from mfg_finance.models.hft import HFTParams, eta_from_m_alpha, initial_density
from mfg_finance.solver import solve_mfg_picard

grid = Grid1D(-5.0, 5.0, nx=401, T=1.0, nt=200, bc='neumann')
params = HFTParams(nu=0.02, phi=0.1, gamma_T=2.0, eta0=0.5, eta1=0.8, m0_mean=0.0, m0_std=1.0)

m0 = initial_density(grid, params)
U_all, M_all, alpha_all, errors, metrics = solve_mfg_picard(
    grid,
    params,
    max_iter=100,
    tol=1e-6,
    mix=0.2,
    m0=m0,
    eta_callback=eta_from_m_alpha,
)

print('iterations:', len(errors))
print('final_error:', errors[-1] if errors else None)
print('errors first 10:', errors[:10])
print('errors last 5:', errors[-5:])
