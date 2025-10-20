# Mean Field Games para o mercado brasileiro
Solver numerico de **Mean Field Games (MFG)** em 1D aplicado a microestrutura da B3. O sistema acopla **Hamilton–Jacobi–Bellman (HJB)** e **Fokker–Planck (FP)** resolvidos por iteracao de Picard com esquemas conservativos (Lax-Friedrichs + upwind). O projeto oferece CLI, notebook, scripts de preparacao de dados e testes automatizados.

## Visao geral
O modelo conecta decisoes individuais de agentes de alta frequencia a efeitos agregados (campo medio). Cada agente decide esforcos de negociacao para minimizar custos de execucao e inventario, enquanto a media das decisoes retroalimenta o ambiente enfrentado por todos. O solver busca o equilibrio alternando HJB (valor) e FP (densidade) com amortecimento adaptativo.

## Equacoes (visao rapida)
**HJB (backward)**

$$
\begin{cases}
& -\partial_t U(t,x) - 
u \Delta U(t,x) + H(
abla U(t,x), m(t,x)) = 0 \
& U(T,x) = \gamma_T x^2
\end{cases}
$$

**FP (forward)**

$$
egin{aligned}
& \partial_t m(t,x) - 
u \Delta m(t,x) - 
abla\cdotig(m(t,x)v(t,x)ig) = 0 \
& m(0,x) = m_0(x)
\end{aligned}
$$

**Controle otimo LQ**

$$
egin{aligned}
& lpha^{*}(t,x) = -rac{\partial_x U(t,x)}{\eta(m)} \
& \eta(m) = \eta_0 + \eta_1 \lvert \overline{lpha} 
vert
\end{aligned}
$$

> **1D:** $
abla U \equiv \partial_x U$ e $
abla\cdot(mv) \equiv \partial_x(mv)$.

## Pipeline visual
![Distribuicao](notebooks_output/run-20251020-150052/density_small.png)
*Distribuicao do FP ao longo do tempo; a massa permanece conservada.*

![Funcao valor](notebooks_output/run-20251020-150052/value_function_small.png)
*Funcao valor do HJB mostrando o custo futuro e o impacto das bordas.*

![Politica otima](notebooks_output/run-20251020-150052/alpha_cuts_small.png)
*Quatro cortes da politica otima $alpha(t,x)$ evidenciam o alisamento do controle.*

![Convergencia](notebooks_output/run-20251020-150052/convergence_small.png)
*Erro $L^2$ entre iteracoes Picard; a queda monotonica confirma estabilidade.*

![Preco endogeno](notebooks_output/run-20251020-150052/price_small.png)
*Trajetoria do preco de clearing calibrado para media quase nula; oscilacoes refletem a oferta empirica.*

Para reproduzir o painel sem abrir o Jupyter execute `python scripts/run_notebook_pipeline.py`.

## Interpretacao de mercado (dados 1986–2025)
- O clearing encontra preco medio praticamente zero (`price_mean ~ 4e-05` e faixa total ~1.9e-04), indicando que as curvas de oferta e demanda agregadas se equilibram sem empurrar o mercado.
- O controle otimo e muito suave (`|alpha| medio ~ 0.01`), sinal de que os custos calibrados desencorajam fluxos agressivos e que a liquidez agregada absorve ordens com facilidade.
- A densidade de inventario fica concentrada no centro; agentes retornam rapidamente a posicoes neutras, coerente com penalidades terminais altas (`gamma_T = 0.146815`).
- A calibracao empirica resulta em parametros: `nu = 5.2e-05`, `phi = 2.09e-04`, `eta0 = 1.0e-04`, `eta1 = 7.1428e-02`. Esses valores refletem spreads historicos maiores quando incorporamos COTAHIST 1986–2025.

## Instalacao
```bash
git clone https://github.com/<org>/mfg-for-financial-market.git
cd mfg-for-financial-market
python -m venv .venv && . .venv/Scripts/activate  # Windows
# source .venv/bin/activate                       # macOS/Linux
pip install -e .[dev]
PYTHONPATH=src python -m pytest -q                # smoke opcional
```
> **Notebook**: se nao instalar o pacote, garanta que `src/` esteja no `sys.path`. A primeira celula de `notebooks/mfg_pipeline.ipynb` ja faz esse ajuste.

## Como rodar
```bash
# baseline com clearing endogeno
python -m mfg_finance.cli run --config configs/baseline.yaml --endogenous-price

# sweep de parametros (phi x gamma_T)
python -m mfg_finance.cli sweep   --config configs/baseline.yaml   --phi 0.02,0.035359,0.05   --gamma_T 0.4,2.778412
```
Artefatos vao para `artifacts/run-YYYYmmdd-HHMMSS/` ou `artifacts/sweep-.../` com arrays (`*.npy`), metricas (`metrics.json`), curvas de preco (`price.csv`) e figuras (`*.png`).

## Ajustes finos
- `mix`, `mix_min`, `mix_decay`, `stagnation_tol`: controlam o amortecimento do Picard.
- `relative_tol`: criterio relativo adicional (alem do `tol` absoluto) para encerrar o laco.
- `hjb_inner` / `hjb_tol`: esforco interno do solver HJB.
- `solver.supply` e `solver.price_sensitivity`: curva empirica de oferta e sensibilidade de clearing (ver proxima secao). O baseline usa `price_sensitivity = 30.0`, obtendo preco medio ~0.

### Metricas salvas
`metrics.json` inclui:
- `final_error`, `final_error_relative`, `iterations`
- `mix_history`, `relative_errors`
- `mean_abs_alpha`, `std_alpha`, `liquidity_proxy`
- `price_mean`, `price_std`, `price_min`, `price_max`, `price_span`

## Dados e reproducao
1. **Ingestao COTAHIST (1986–2025):** posicione os arquivos anuais em `data/b3/` e rode:
   ```bash
   python scripts/ingest_cotahist.py
   python scripts/ingest_cotahist_equities.py
   ```
2. **Curva de oferta e calibracao empirica:**
   ```bash
   python scripts/calib_empirical.py
   python scripts/update_solver_config.py --scale 5e-05 --price-sensitivity 30.0
   ```
   Isso atualiza `data/processed/supply_curve.csv` e grava os parametros na `configs/baseline.yaml`.
3. **Painel notebook:** `python scripts/run_notebook_pipeline.py` gera `notebooks_output/run-YYYYmmdd-HHMMSS/`.

> **Aviso legal:** dados COTAHIST pertencem a B3. Certifique-se de possuir licenca antes de utiliza-los.

## Testes
```bash
PYTHONPATH=src python -m pytest -q
```
Os testes cobrem conservacao de massa, positividade, convergencia Picard e refinamento de malha.

## Estrutura do repositorio
```
configs/                  # YAMLs reprodutiveis (baseline, sweeps)
docs/                     # documentacao adicional (ex.: DATA.md)
data/                     # insumos brutos/derivados (nao versionados)
examples/                 # scripts de experimentos rapidos
notebooks/                # notebooks exploratorios
notebooks_output/         # artefatos do pipeline/notebook
reports/                  # figuras e relatorios HTML/PNG
scripts/                  # utilidades de dados e pipeline
src/mfg_finance/          # implementacao do solver
tests/                    # suite PyTest
```

## Roadmap
- Acomodar modelos com ruido comum (SPDE).
- Implementar policy iteration / Newton para aceleracao.
- Preco endogeno via mecanismos de clearing alternativos.
- Extensoes 2D e problemas nao quadraticos.


