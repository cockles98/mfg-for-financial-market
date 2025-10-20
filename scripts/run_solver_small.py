from mfg_finance.grid import Grid1D
from mfg_finance.models.hft import HFTParams, eta_from_m_alpha, initial_density
from mfg_finance.solver import solve_mfg_picard

grid = Grid1D(-1.0, 1.0, nx=81, T=0.5, nt=40, bc='neumann')
params = HFTParams(nu=0.05, phi=0.05, gamma_T=0.5, eta0=0.4, eta1=0.4, m0_mean=0.0, m0_std=0.8)

m0 = initial_density(grid, params)
U_all, M_all, alpha_all, errors, metrics = solve_mfg_picard(
    grid,
    params,
    max_iter=80,
    tol=1e-5,
    mix=0.5,
    m0=m0,
    eta_callback=eta_from_m_alpha,
)

print('iterations:', len(errors))
print('final_error:', errors[-1] if errors else None)
print('mean_abs_alpha:', metrics['mean_abs_alpha'])
