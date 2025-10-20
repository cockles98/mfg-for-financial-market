# Mean Field Game Theory na bolsa brasileira
Solver de **Mean Field Games (MFG)** para finanças em 1D, acoplando **Hamilton–Jacobi–Bellman (HJB)** e **Fokker–Planck (FP)** com iteração de Picard, Lax-Friedrichs no HJB e upwind conservativo no FP. O projeto inclui CLI, experimentos reprodutíveis, métricas e testes de massa/positividade/convergência.

Resumidamente, o projeto conecta otimização individual e efeitos de multidão no mercado. Em vez de modelar um trader isolado, usa-se a estrutura de Mean Field Games (MFG): cada agente escolhe suas ações para minimizar custos (por exemplo, custo de execução e carregar inventário), enquanto a média das escolhas afeta o ambiente que todos enfrentam.

### **O que o código faz**
- Resolve duas equações acopladas no tempo:
  - HJB (decisão ótima): calcula o “valor” de cada estado e a política ótima de negociação.
  - Fokker-Planck (população): descreve como a distribuição de posições dos agentes evolui.
- Encontra o equilíbrio por um laço de ponto-fixo (Picard), alternando HJB (para trás no tempo) e FP (para frente) até convergir.
- Usa esquemas numéricos estáveis reconhecidos na literatura: Lax-Friedrichs (gradiente monotônico) e upwind conservativo (advecção), com difusão implícita. Isso preserva massa ≈ 1 e impede densidades negativas — requisitos básicos para resultados confiáveis.
-Implementa um caso LQ (quadrático) inspirado em microestrutura/HFT: custo de execução, penalidade de inventário e (opcionalmente) custo dependente do fluxo médio do grupo.

### **Por que isso importa**
Esse arranjo permite experimentar hipóteses de mercado de forma controlada: como a liquidez muda quando negociar fica mais caro? O grupo tende a carregar mais ou menos inventário? A política ótima fica mais agressiva ou mais cautelosa? Os gráficos e métricas ajudam a visualizar esses regimes.

## Destaques
- 🔁 **HJB ↔ FP** com laço de **Picard** e *under-relaxation*.
- 🧮 **Esquemas numéricos estáveis**: Lax-Friedrichs (grad monotônico) e upwind conservativo (advecção), difusão implícita via SciPy sparse.
- 📈 **Modelo HFT LQ** (inventário + custo de execução endógeno opcional).
- 🧪 **Testes**: conservação de massa, positividade, convergência do Picard e **refinamento de malha**.
- 🧰 **CLI** para rodar *baseline*, varrer parâmetros e salvar artefatos (figuras, `.npy`, `metrics.json`, `summary.csv`).
- 🗺️ **Config YAML** para reprodutibilidade.

## Equações (visão rápida)
**HJB (backward)**

$$
-\partial_t U(t,x)\;-\;\nu\,\Delta U(t,x)\;+\;H\!\big(\nabla U(t,x),\,m(t,x)\big)\;=\;0,
\quad
U(T,x)=\gamma_T x^2.
$$

**FP (forward)**

$$
\partial_t m(t,x)\;-\;\nu\,\Delta m(t,x)\;-\;\nabla\!\cdot\!\big(m(t,x)\,v(t,x)\big)\;=\;0,
\quad
m(0,x)=m_0(x).
$$

**Controle ótimo LQ**

$$
\alpha^{*}(t,x)\;=\;-\frac{\partial_x U(t,x)}{\eta(m)},
\quad
\eta(m)=\eta_0+\eta_1\,\big|\overline{\alpha}\big|.
$$

> **Nota:** Em 1D, use $\nabla U \equiv \partial_x U$ e $\nabla\!\cdot(mv)\equiv \partial_x(mv)$.

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
