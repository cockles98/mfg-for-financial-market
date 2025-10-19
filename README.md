# mfg-finance
**Solver de Mean Field Games (MFG) para finanças** em 1D, acoplando **Hamilton–Jacobi–Bellman (HJB)** e **Fokker–Planck (FP)** com **iteração de Picard**, **Lax-Friedrichs** no HJB e **upwind conservativo** no FP. O projeto inclui CLI, experimentos reprodutíveis, métricas e testes de massa/positividade/convergência.

## Descrição curta (About)
> Mean Field Games para finanças: solver 1D HJB–Fokker-Planck com iteração de Picard, esquema monotônico (Lax-Friedrichs), upwind conservativo e varreduras de parâmetros.

## Destaques
- 🔁 **HJB ↔ FP** com laço de **Picard** e *under-relaxation*.
- 🧮 **Esquemas numéricos estáveis**: Lax-Friedrichs (grad monotônico) e upwind conservativo (advecção), difusão implícita via SciPy sparse.
- 📈 **Modelo HFT LQ** (inventário + custo de execução endógeno opcional).
- 🧪 **Testes**: conservação de massa, positividade, convergência do Picard e **refinamento de malha**.
- 🧰 **CLI** para rodar *baseline*, varrer parâmetros e salvar artefatos (figuras, `.npy`, `metrics.json`, `summary.csv`).
- 🗺️ **Config YAML** para reprodutibilidade.

## Equações (visão rápida)
- **HJB (backward)**: \(-\partial_t U - \nu \Delta U + H(\nabla U, m) = 0\), \(U(T,x)=\gamma_T x^2\)  
- **FP (forward)**: \(\partial_t m - \nu \Delta m - \nabla\cdot(m\,v)=0\), \(m(0,x)=m_0(x)\)  
- **Controle ótimo LQ**: \(\alpha^{\*} = -\partial_x U / \eta(m)\), com \(\eta(m)=\eta_0+\eta_1\,|\overline{\alpha}|\)

## Requisitos
Python ≥ 3.10 · `numpy` · `scipy` · `matplotlib` · `pyyaml` · `tqdm` · `pytest`

## Instalação
```bash
# clone
git clone https://github.com/<org>/mfg-finance.git
cd mfg-finance

# (opcional) criar venv
python -m venv .venv && . .venv/Scripts/activate  # Windows
# source .venv/bin/activate                       # macOS/Linux

# instalar deps
pip install -e .[dev]
```

## Como rodar
```bash
# experimento baseline
python -m mfg_finance.cli run --config configs/baseline.yaml

# varredura de parâmetros (exemplo)
python -m mfg_finance.cli sweep --phi 0.05,0.1,0.2 --gamma_T 1.0,2.0
```
Saídas ficam em `artifacts/run-YYYYmmdd-HHMMSS/` (figuras `.png`, arrays `.npy`, `metrics.json`, `summary.csv`).

## Testes e validações
```bash
pytest -q
```
- **Massa ≈ 1** ao longo do tempo  
- **Positividade** de `m` (pós-projeção)  
- **Convergência do Picard** (erro decrescente)  
- **Refinamento de malha** (norma entre soluções diminui com `nx↑, nt↑`)

## Estrutura (resumo)
```
src/mfg_finance/
  grid.py        # grade e BCs
  ops.py         # laplaciano, grad, upwind, utilitários
  hamiltonian.py # H, alpha*, custos LQ
  hjb.py         # passo backward (Lax-Friedrichs)
  fp.py          # passo forward (upwind + difusão implícita)
  solver.py      # laço de Picard + métricas
  models/hft.py  # parâmetros e densidade inicial
  viz.py         # plots (densidade, valor, alpha, convergência)
  cli.py         # interface de linha de comando
configs/baseline.yaml
tests/...
```

## Roadmap
- Ruído comum / SPDE
- Policy iteration / Newton
- Preço endógeno por *clearing* (opcional no CLI)
- Extensão 2D e casos não-quadráticos

## Licença
MIT (sugestão).
