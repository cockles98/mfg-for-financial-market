# Mean Field Games para o mercado brasileiro
Solver numérico de **Mean Field Games (MFG)** em 1D aplicado à microestrutura da B3. O sistema acopla **Hamilton–Jacobi–Bellman (HJB)** e **Fokker–Planck (FP)** resolvidos por iteração de Picard com esquemas conservativos (Lax-Friedrichs + upwind). O projeto oferece CLI, notebook, scripts de preparação de dados e testes automatizados.

## Visão geral
O modelo conecta decisões individuais de agentes de alta frequência a efeitos agregados (campo médio). Cada agente decide esforços de negociação para minimizar custos de execução e inventário, enquanto a média das decisões retroalimenta o ambiente enfrentado por todos. O solver busca o equilíbrio alternando HJB (valor) e FP (densidade) com amortecimento adaptativo.

## Equações (visão rápida)
**HJB (backward)**

$$
egin{aligned}
& -\partial_t U(t,x) - 
u \Delta U(t,x) + H(
abla U(t,x), m(t,x)) = 0 \
& U(T,x) = \gamma_T x^2
\end{aligned}
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

**Controle ótimo LQ**

$$
egin{aligned}
& lpha^{*}(t,x) = -rac{\partial_x U(t,x)}{\eta(m)} \
& \eta(m) = \eta_0 + \eta_1 \lvert \overline{lpha} vert
\end{aligned}
$$

> **1D:** $
abla U \equiv \partial_x U$ e $
abla\cdot(mv) \equiv \partial_x(mv)$.

## Pipeline visual
![Distribuição](notebooks_output/run-20251020-005200/density_small.png)
*Distribuição do FP ao longo do tempo; a massa permanece conservada.*

![Função valor](notebooks_output/run-20251020-005200/value_function_small.png)
*Função valor do HJB mostrando o custo futuro e o impacto das bordas.*

![Política ótima](notebooks_output/run-20251020-005200/alpha_cuts_small.png)
*Quatro cortes da política ótima $lpha(t,x)$ evidenciam o alisamento do controle.*

![Convergência](notebooks_output/run-20251020-005200/convergence_small.png)
*Erro $L^2$ entre iterações Picard; a queda monotônica confirma estabilidade.*

![Preço endógeno](notebooks_output/run-20251020-005200/price_small.png)
*Trajetória do preço de clearing calibrado para média quase nula; oscilações refletem a oferta empírica.*

Para reproduzir o painel sem abrir o Jupyter execute `python scripts/run_notebook_pipeline.py`.

## Instalação
```bash
git clone https://github.com/<org>/mfg-for-financial-market.git
cd mfg-for-financial-market
python -m venv .venv && . .venv/Scripts/activate  # Windows
# source .venv/bin/activate                       # macOS/Linux
pip install -e .[dev]
PYTHONPATH=src python -m pytest -q                # smoke opcional
```
> **Notebook**: se não instalar o pacote, garanta que `src/` esteja no `sys.path`. A primeira célula de `notebooks/mfg_pipeline.ipynb` já faz esse ajuste.

## Como rodar
```bash
# baseline com clearing endógeno
python -m mfg_finance.cli run --config configs/baseline.yaml --endogenous-price

# sweep de parâmetros (phi x gamma_T)
python -m mfg_finance.cli sweep   --config configs/baseline.yaml   --phi 0.02,0.035359,0.05   --gamma_T 0.4,0.568862
```
Artefatos vão para `artifacts/run-YYYYmmdd-HHMMSS/` ou `artifacts/sweep-.../` com arrays (`*.npy`), métricas (`metrics.json`), curvas de preço (`price.csv`) e figuras (`*.png`).

## Ajustes finos
- `mix`, `mix_min`, `mix_decay`, `stagnation_tol`: controlam o amortecimento do Picard.
- `relative_tol`: critério relativo adicional (além do `tol` absoluto) para encerrar o laço.
- `hjb_inner` / `hjb_tol`: esforço interno do solver HJB.
- `solver.supply` e `solver.price_sensitivity`: curva empírica de oferta e sensibilidade de clearing (ver seção “Dados”). O baseline usa `price_sensitivity = 30.0`, obtendo preço médio ≈ 0.

### Métricas salvas
`metrics.json` inclui:
- `final_error`, `final_error_relative`, `iterations`
- `mix_history`, `relative_errors`
- `mean_abs_alpha`, `std_alpha`, `liquidity_proxy`
- `price_mean`, `price_std`, `price_min`, `price_max`, `price_span` (quando o clearing roda)

## Dados e reprodução
1. **Ingestão COTAHIST** (não versionada): copie os arquivos originais para `data/raw/` e use os scripts em `scripts/` para gerar os Parquets de `data/processed/` (detalhes em `docs/DATA.md`).
2. **Curva de oferta** (quantis de volume/spread):
   ```bash
   python scripts/build_supply_curve.py      --input data/processed/cotahist_equities_extended.parquet      --output data/processed/supply_curve.csv
   ```
3. **Atualize o baseline** com a curva e a sensibilidade desejada:
   ```bash
   python scripts/update_solver_config.py      --supply data/processed/supply_curve.csv      --config configs/baseline.yaml      --scale 5e-05      --price-sensitivity 30.0
   ```
4. (Opcional) rode `python scripts/run_notebook_pipeline.py` para gerar `notebooks_output/run-YYYYmmdd-HHMMSS/`.

> **Aviso legal:** dados COTAHIST pertencem à B3. Certifique-se de possuir licença antes de utilizá-los.

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
notebooks_output/         # artefatos do pipeline/notebook
reports/                  # figuras e relatórios HTML/PNG
scripts/                  # utilidades de dados e pipeline
src/mfg_finance/          # implementação do solver
tests/                    # suíte PyTest
```

## Roadmap
- Acomodar modelos com ruído comum (SPDE).
- Implementar policy iteration / Newton para aceleração.
- Preço endógeno via mecanismos de clearing alternativos.
- Extensões 2D e problemas não quadráticos.
