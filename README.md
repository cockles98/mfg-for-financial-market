# mfg-finance

Academic prototype (undergraduate research) that blends real B3 data with a linear-quadratic Mean Field Game (MFG). The goal is to show how thousands of similar agents manage inventory while the market reacts to aggregate order flow.

## Model summary

- Agents control an order flow `alpha` and prefer inventories close to zero.
- Execution friction depends on the average flow: `eta(m) = eta0 + eta1 * |mean(alpha)|`.
- Quadratic costs: running (`phi`) and terminal (`gamma_T`) penalties.
- Coupled HJB / Fokker-Planck equations solved via Picard iteration.
- Optional endogenous price computed by instantaneous clearing.

## LQ assumptions and current limitations

- Only idiosyncratic noise (no common noise, no dominant agent).
- One-dimensional state (inventory only); no spreads or extra factors.
- Picard iteration with numerical safeguards; dense grids still require acceleration (Newton/policy iteration).
- Execution costs and supply are heuristic; calibration with intraday volume/spread remains future work.

## Numerical choices / guarantees

- HJB: Lax-Friedrichs with bounded dissipation plus relaxed implicit step.
- FP: upwind conservative advection plus implicit diffusion.
- Positivity and mass conservation enforced (projection on the simplex).
- Tests cover operators, Picard stability, mesh refinement.

## Installation

```bash
pip install -e .
```

Python >= 3.10 with packages from `pyproject.toml` (`numpy`, `scipy`, `matplotlib`, `pyyaml`, `tqdm`, `pytest`).

## Quick start

- **CLI**
  ```bash
  python -m mfg_finance.cli run --config configs/baseline.yaml
  ```
  Saves artefacts under `artifacts/run-*/`. Add `--endogenous-price` to compute the clearing price.

- **Notebook**: `notebooks/mfg_pipeline.ipynb` calibrates, runs the solver, and writes outputs to `notebooks_output/`.
- **Scripts**: `examples/run_baseline.py` and `examples/run_sweep_phi_gamma.py` reproduce standard scenarios.

## Repository structure and status

| Path | Purpose | Status |
|------|---------|--------|
| `src/mfg_finance/` | Grid, operators, HJB/FP, Picard, price, viz, CLI | OK (with safeguards) |
| `data/processed/`  | Clean COTAHIST parquet/CSV, returns/vol stats | OK (ready to use) |
| `configs/baseline.yaml` | Moderate grid (201x150) + soft penalties | OK (convergent) |
| `tests/` | `pytest -q` covers operators, mass, Picard, refinement | OK (all green) |
| `reports/analytics/` | Aggregated charts (histograms, top vol) | OK (auto-generated) |
| `reports/final/mfg_report.html` | Plain-language summary + figures | OK (ready for presentation) |

### Current state (academic focus)
- Demonstration only; not a production pricing engine.
- Heuristics match the order-of-magnitude for volatility/intensity, but execution costs and supply still lack calibration with real volume/spread data.
- Numerical safeguards (gradient/value caps, relaxation) keep the solver stable on moderate grids; larger grids still need faster methods.
- Pipeline already produces a complete report in `reports/final/`.

## Tests

```bash
pytest -q
```

18 tests cover operators, solver stability, mass conservation, mesh refinement, and Picard.

## Immediate roadmap

1. **Empirical calibration**: fit `eta0`, `eta1`, `phi`, `gamma_T` using intraday volume/spread or order-book data.
2. **Numerical acceleration**: Newton/policy iteration and/or adaptive meshes for larger grids.
3. **Common noise / heterogeneous agents**: add systemic shocks and a large player (market maker vs. crowd) to compare liquidity regimes.

## License

MIT
