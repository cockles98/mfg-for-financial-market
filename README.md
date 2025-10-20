# Mean Field Game Theory na bolsa brasileira
Solver de **Mean Field Games (MFG)** para finanÃ§as em 1D, acoplando **Hamiltonâ€“Jacobiâ€“Bellman (HJB)** e **Fokkerâ€“Planck (FP)** com iteraÃ§Ã£o de Picard, Lax-Friedrichs no HJB e upwind conservativo no FP. O projeto inclui CLI, experimentos reprodutÃ­veis, mÃ©tricas e testes de massa/positividade/convergÃªncia.

Resumidamente, o projeto conecta otimizaÃ§Ã£o individual e efeitos de multidÃ£o no mercado. Em vez de modelar um trader isolado, usa-se a estrutura de Mean Field Games (MFG): cada agente escolhe suas aÃ§Ãµes para minimizar custos (por exemplo, custo de execuÃ§Ã£o e carregar inventÃ¡rio), enquanto a mÃ©dia das escolhas afeta o ambiente que todos enfrentam.

### **O que o cÃ³digo faz**
- Resolve duas equaÃ§Ãµes acopladas no tempo:
  - HJB (decisÃ£o Ã³tima): calcula o â€œvalorâ€ de cada estado e a polÃ­tica Ã³tima de negociaÃ§Ã£o.
  - Fokker-Planck (populaÃ§Ã£o): descreve como a distribuiÃ§Ã£o de posiÃ§Ãµes dos agentes evolui.
- Encontra o equilÃ­brio por um laÃ§o de ponto-fixo (Picard), alternando HJB (para trÃ¡s no tempo) e FP (para frente) atÃ© convergir.
- Usa esquemas numÃ©ricos estÃ¡veis reconhecidos na literatura: Lax-Friedrichs (gradiente monotÃ´nico) e upwind conservativo (advecÃ§Ã£o), com difusÃ£o implÃ­cita. Isso preserva massa â‰ˆ 1 e impede densidades negativas â€” requisitos bÃ¡sicos para resultados confiÃ¡veis.
-Implementa um caso LQ (quadrÃ¡tico) inspirado em microestrutura/HFT: custo de execuÃ§Ã£o, penalidade de inventÃ¡rio e (opcionalmente) custo dependente do fluxo mÃ©dio do grupo.

### **Por que isso importa**
Esse arranjo permite experimentar hipÃ³teses de mercado de forma controlada: como a liquidez muda quando negociar fica mais caro? O grupo tende a carregar mais ou menos inventÃ¡rio? A polÃ­tica Ã³tima fica mais agressiva ou mais cautelosa? Os grÃ¡ficos e mÃ©tricas ajudam a visualizar esses regimes.

## Destaques
- ðŸ” **HJB â†” FP** com laÃ§o de **Picard** e *under-relaxation*.
- ðŸ§® **Esquemas numÃ©ricos estÃ¡veis**: Lax-Friedrichs (grad monotÃ´nico) e upwind conservativo (advecÃ§Ã£o), difusÃ£o implÃ­cita via SciPy sparse.
- ðŸ“ˆ **Modelo HFT LQ** (inventÃ¡rio + custo de execuÃ§Ã£o endÃ³geno opcional).
- ðŸ§ª **Testes**: conservaÃ§Ã£o de massa, positividade, convergÃªncia do Picard e **refinamento de malha**.
- ðŸ§° **CLI** para rodar *baseline*, varrer parÃ¢metros e salvar artefatos (figuras, `.npy`, `metrics.json`, `summary.csv`).
- ðŸ—ºï¸ **Config YAML** para reprodutibilidade.

## EquaÃ§Ãµes (visÃ£o rÃ¡pida)

**HJB (backward)**

$$
\begin{cases}
& -\partial_t U(t,x) - \nu \Delta U(t,x) + H(\nabla U(t,x), m(t,x)) = 0 \\
& U(T,x) = \gamma_T x^2
\end{cases}
$$

**FP (forward)**

$$
\begin{cases}
& \partial_t m(t,x) - \nu \Delta m(t,x) - \nabla\cdot\big(m(t,x)v(t,x)\big) = 0 \\
& m(0,x) = m_0(x)
\end{cases}
$$

**Controle Ã³timo LQ**

$$
\begin{cases}
& \alpha^{*}(t,x) = -\frac{\partial_x U(t,x)}{\eta(m)} \\
& \eta(m) = \eta_0 + \eta_1 \lvert \overline{\alpha} \rvert
\end{cases}
$$

> **1D:** $\nabla U \equiv \partial_x U$ e $\nabla\cdot(mv)\equiv \partial_x(mv)$.

## Requisitos
Python â‰¥ 3.10 Â· `numpy` Â· `scipy` Â· `matplotlib` Â· `pyyaml` Â· `tqdm` Â· `pytest`

## InstalaÃ§Ã£o
```bash
# clone
git clone https://github.com/<org>/mfg-finance.git
cd mfg-finance

# (opcional) criar venv
python -m venv .venv && . .venv/Scripts/activate  # Windows
# source .venv/bin/activate                       # macOS/Linux

# instalar deps
pip install -e .[dev]

# (opcional) rodar testes rÃ¡pidos para validar setup
PYTHONPATH=src python -m pytest -q
```

> **Dica notebooks**: se nÃ£o quiser instalar o pacote, adicione `src/` ao `sys.path`. O notebook `notebooks/mfg_pipeline.ipynb` jÃ¡ inclui esse *bootstrap* nas primeiras cÃ©lulas.

## Como rodar
```bash
# experimento baseline
python -m mfg_finance.cli run --config configs/baseline.yaml

# varredura de parÃ¢metros (exemplo)
python -m mfg_finance.cli sweep --phi 0.02,0.035359,0.05 --gamma_T 0.4,0.568862
```
SaÃ­das ficam em `artifacts/run-YYYYmmdd-HHMMSS/` (figuras `.png`, arrays `.npy`, `metrics.json`, `summary.csv`).

### Ajustes finos do solver
- `mix`, `mix_min`, `mix_decay` e `stagnation_tol` controlam a *under-relaxation* adaptativa do laço de Picard.
- `relative_tol` encerra a iteração quando a variação relativa da densidade cai abaixo do limiar (além do `tol` absoluto tradicional).
- HJB interno aceita `hjb_inner` e `hjb_tol` para limitar iterações internas.
- O clearing endógeno já vem habilitado em `configs/baseline.yaml`: a curva de oferta em `solver.supply` vem do script `scripts/build_supply_curve.py` (COTAHIST 2015-2025) e é normalizada por `scripts/update_solver_config.py`. A sensibilidade padrão `solver.price_sensitivity = 30.0` garante preço médio ≈ 0; ajuste conforme necessidade.
- Para testar regimes alternativos execute `python -m mfg_finance.cli run --config configs/baseline.yaml --endogenous-price`.

### Sweeps rápidos
Use a CLI para varrer parâmetros e validar estabilidade com os dados calibrados:

```bash
python -m mfg_finance.cli sweep ^
  --config configs/baseline.yaml ^
  --phi 0.02,0.035359,0.05 ^
  --gamma_T 0.4,0.568862
```

O comando gera `artifacts/sweep-YYYYmmdd-HHMMSS/summary.csv` com métricas (erro final, |alpha| médio, proxy de liquidez e estatísticas de preço quando o clearing converge). Ajuste as listas conforme o experimento.

### Dados e reprodutibilidade
- Os insumos vêm do **COTAHIST/B3 (2015–2025)**; veja `docs/DATA.md` para detalhes legais e passos de preparação.
- Reconstrua o resumo de oferta com `python scripts/build_supply_curve.py` (exige `data/processed/cotahist_equities_extended.parquet`).
- Propague a curva para o solver rodando `python scripts/update_solver_config.py` (opções `--scale`, `--price-sensitivity` e `--samples` replicam o `baseline`).
- Para gerar os artefatos do notebook sem depender do Jupyter use python scripts/run_notebook_pipeline.py.
- As execuções escrevem `metrics.json` com erros absolutos/relativos e estatísticas de preço (`price_mean`, `price_std`, `price_min`, `price_max`, `price_span`).

## Testes e validaÃ§Ãµes
```bash
pytest -q
```
- **Massa â‰ˆ 1** ao longo do tempo  
- **Positividade** de `m` (pÃ³s-projeÃ§Ã£o)  
- **ConvergÃªncia do Picard** (erro decrescente)  
- **Refinamento de malha** (norma entre soluÃ§Ãµes diminui com `nxâ†‘, ntâ†‘`)

## Estrutura (resumo)
```
configs/                  # YAMLs reproducÃ­veis
docs/                     # notas sobre dados e reprodução
data/                     # insumos brutos e processados
examples/                 # scripts de uso rÃ¡pido
notebooks/                # notebooks exploratÃ³rios
notebooks_output/         # resultados consolidados dos notebooks
reports/                  # figuras e relatÃ³rios finais
scripts/                  # utilidades para limpar/gerar artefatos
src/mfg_finance/
  grid.py                 # grade e BCs
  ops.py                  # laplaciano, grad, upwind, utilitÃ¡rios
  hamiltonian.py          # H, alpha*, custos LQ
  hjb.py                  # passo backward (Lax-Friedrichs)
  fp.py                   # passo forward (upwind + difusÃ£o implÃ­cita)
  solver.py               # laÃ§o de Picard + mÃ©tricas
  models/hft.py           # parÃ¢metros e densidade inicial
  viz.py                  # plots (densidade, valor, alpha, convergÃªncia)
  cli.py                  # interface de linha de comando
tests/
```

## Roadmap
- RuÃ­do comum / SPDE
- Policy iteration / Newton
- PreÃ§o endÃ³geno por *clearing* (opcional no CLI)
- ExtensÃ£o 2D e casos nÃ£o-quadrÃ¡ticos