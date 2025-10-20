# Mean Field Games para o mercado brasileiro
Solver numérico de **Mean Field Games (MFG)** em 1D aplicado à microestrutura da B3. O sistema acopla **Hamilton–Jacobi–Bellman (HJB)** e **Fokker–Planck (FP)** por iteração de Picard com esquemas conservativos (Lax-Friedrichs + upwind). O projeto fornece CLI, notebook, scripts de dados e testes automatizados.

## Visão geral
O modelo liga decisões individuais de agentes de alta frequência a efeitos agregados (“campo médio”). Cada agente escolhe esforços de negociação para minimizar custos de execução e inventário, enquanto a média das escolhas influencia o ambiente enfrentado por todos. O solver busca o equilíbrio fixando valor (HJB) e densidade (FP) em ciclos Picard com amortecimento adaptativo.

### Principais funcionalidades
- **HJB + FP acopladas** com controle LQ e custo endógeno via `eta(m)`.
- **Convergência robusta**: mixagem adaptativa, tolerâncias absoluta/relativa e registro de métricas (`metrics.json`).
- **Dados reais da B3** (COTAHIST 2015–2025) para calibrar volatilidade, curva de oferta e spreads.
- **Ferramentas de reproducibilidade**: scripts para construir `supply_curve.csv`, atualizar `baseline.yaml` e executar o pipeline sem Jupyter.
- **CLI e notebook** para rodar baseline, varrer parâmetros e gerar relatórios completos (`.npy`, `.csv`, `.png`, `metrics.json`).  

## Instalação
```bash
git clone https://github.com/<org>/mfg-for-financial-market.git
cd mfg-for-financial-market
python -m venv .venv && . .venv/Scripts/activate  # Windows
# source .venv/bin/activate                       # macOS/Linux
pip install -e .[dev]
PYTHONPATH=src python -m pytest -q                # smoke opcional
```
> **Notebook**: se não instalar o pacote, basta garantir que `src/` está no `sys.path`. O notebook `notebooks/mfg_pipeline.ipynb` já faz isso na primeira célula.

## Como rodar
```bash
# baseline com clearing endógeno
python -m mfg_finance.cli run --config configs/baseline.yaml --endogenous-price

# sweep de parâmetros (φ x γ_T)
python -m mfg_finance.cli sweep \
  --config configs/baseline.yaml \
  --phi 0.02,0.035359,0.05 \
  --gamma_T 0.4,0.568862
```
Artefatos são gerados em `artifacts/run-YYYYmmdd-HHMMSS/` ou `artifacts/sweep-.../` com arrays (`*.npy`), métricas (`metrics.json`), curvas de preço (`price.csv`) e figuras (`*.png`).

### Ajustes finos
- `mix`, `mix_min`, `mix_decay`, `stagnation_tol`: controlam o amortecimento do Picard.
- `relative_tol`: critério relativo adicional (além do `tol` absoluto) para parar o laço.
- `hjb_inner` / `hjb_tol`: esforço interno do solver HJB.
- `solver.supply` e `solver.price_sensitivity`: curva empírica de oferta e sensibilidade de clearing (ver seção “Dados”). O baseline usa `price_sensitivity = 30.0`, resultando em preço médio ≈ 0.
- Para gerar os artefatos do notebook sem abrir o Jupyter:  
  `python scripts/run_notebook_pipeline.py`

### Métricas
`metrics.json` (CLI/notebook) inclui:
- `final_error`, `final_error_relative`, `iterations`
- `mix_history`, `relative_errors`
- `mean_abs_alpha`, `std_alpha`, `liquidity_proxy`
- `price_mean`, `price_std`, `price_min`, `price_max`, `price_span` (quando o clearing roda)

## Dados e reprodução
1. **Ingestão COTAHIST** (não versionada): copie os arquivos originais para `data/raw/` e use os scripts da pasta `scripts/` para gerar os Parquets em `data/processed/` (ver `docs/DATA.md`).  
2. **Curva de oferta** (volume/spread por quantil):
   ```bash
   python scripts/build_supply_curve.py \
     --input data/processed/cotahist_equities_extended.parquet \
     --output data/processed/supply_curve.csv
   ```
3. **Atualizar baseline** com a curva e sensibilidade desejada:
   ```bash
   python scripts/update_solver_config.py \
     --supply data/processed/supply_curve.csv \
     --config configs/baseline.yaml \
     --scale 5e-05 \
     --price-sensitivity 30.0
   ```
4. Opcional, executar `scripts/run_notebook_pipeline.py` para replicar o notebook completo e salvar os artefatos em `notebooks_output/run-YYYYmmdd-HHMMSS/`.

> **Aviso legal:** dados COTAHIST pertencem à B3 e não são redistribuídos aqui. Certifique-se de possuir licença antes de executar os scripts.

## Testes
```bash
PYTHONPATH=src python -m pytest -q
```
Os testes cobrem conservação de massa, positividade, convergência Picard e refinamento de malha.

## Estrutura do repositório
```
configs/                  # YAMLs reprodutíveis (baseline, sweeps)
docs/                     # documentação adicional (ex.: DATA.md)
data/                     # insumos brutos/derivados (não versionados)
examples/                 # scripts de experimentos rápidos
notebooks/                # notebooks exploratórios
notebooks_output/         # artefatos gerados pelo pipeline
reports/                  # figuras e relatórios HTML/PNG
scripts/                  # utilidades (dados, limpeza, pipeline)
src/mfg_finance/          # implementação do solver
tests/                    # suíte PyTest
```

## Roadmap
- Acomodar modelos com ruído comum (SPDE).
- Implementar policy iteration / Newton para aceleração.
- Preço endógeno via mecanismo de clearing alternativo.
- Extensões 2D e problemas não quadráticos.
