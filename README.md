# mfg-finance

Implementacao em Python de um jogo de campo medio (Mean Field Game, MFG) para microestrutura de mercado. O modelo acopla a equacao de Hamilton-Jacobi-Bellman (HJB) e a equacao de Fokker-Planck (FP) por meio de uma iteracao de Picard, permitindo estudar as estrategias de negociacao de agentes identicos em continuo.

## Visao geral do MFG

- Agentes atomisticos controlam fluxos de ordem (lpha) para gerir inventario.
- A friccao de execucao depende do fluxo medio: eta(m) = eta0 + eta1 * |mean(alpha)|.
- O terminal payoff penaliza inventario com peso gamma_T, enquanto phi pune inventario corrente.
- Opcionalmente, um preco endogeno P(t) pode ser calculado por clearing instantaneo quando habilitado.

## Hipoteses LQ

- Custos quadraticos (LQ) em inventario e controle; gradientes produzem controles lineares.
- Ruido apenas idiossincratico (
u), sem componente comum ou agente dominante.
- Estado unidimensional (inventario); nao ha restricoes rigidas de caixa ou alavancagem.

## Escolhas numericas

- **HJB**: esquema Lax-Friedrichs monotono (dissipacao adaptativa) seguido de resolucao implicita difusiva.
- **FP**: adveccao upwind conservativa explicita + difusao implicita (semi-implicito).
- **Iteracao**: Picard com relaxacao mix, tolerancia L2 (||M^{k+1} - M^k||_2).
- **CFL heuristico**: Grid1D.suggest_cfl_limits() informa limites difusivos/adveccao e o CLI emite avisos quando dt e agressivo.

## Garantias numericas

- **Positividade**: densidades projetadas para o simplex (corte em zero + renormalizacao).
- **Conservacao de massa**: FP semi-implicito preserva sum m dx ≈ 1 (testado automaticamente).
- **Refinamento**: testes verificam que a norma L2 entre solucao grossa/fina diminui quando se dobra a malha.

## Limitacoes atuais

- Dominio 1D; nao ha dinamica multi-ativo nem preco resiliente multi-dimensional.
- Ausencia de ruido comum, agente dominante ou heterogeneidade estrutural.
- Picard pode ser lento em regimes rigidos; nao ha Newton/policy iteration.
- Sem calibracao empirica ou validacao com dados historicos.

## Instalação

`ash
pip install -e .
`

Requisitos: Python >= 3.10 e pacotes listados em pyproject.toml (
umpy, scipy, matplotlib, pyyaml, 	qdm, pytest).

## Execucao de experiencias

### CLI principal

`ash
python -m mfg_finance.cli run --config configs/baseline.yaml
`

- Artefatos em rtifacts/run-*/ (U_all.npy, M_all.npy, lpha_all.npy, metrics.json, figuras, etc.).
- Para preco endogeno instantaneo adicione --endogenous-price (gera price.csv e price.png).

### Script baseline

`ash
python -m examples.run_baseline
`

Salva matrizes e figuras (density.png, alue.png, lpha_cuts.png, convergence.png) em examples_output/.

### Varredura phi/gamma_T

`ash
python -m examples.run_sweep_phi_gamma
`

Gera examples_output/sweep_phi_gamma/ com heatmap_alpha_mean.png, sweep_summary.csv e matrizes auxiliares. Evidencia a transicao de liquidez quando phi/gamma_T aumenta (round-tripper -> small-IT).

## Testes

`ash
pytest -q
`

18 testes cobrem operadores, HJB/FP, Picard, refinamento de malha e suites auxiliares.

## Estrutura do projeto

- src/mfg_finance/: grid, operadores, Hamiltoniana LQ, resolvers HJB/FP, Picard, precificacao, visualizacao e CLI.
- configs/baseline.yaml: configuracao principal (malha 401x200, phi=0.1, gamma_T=2.0).
- 	ests/: regressao numerica (mass, convergencia, refinamento).
- examples/: scripts de reproducao (baseline e sweep).

## Roadmap

1. **Ruido comum** e choques sistemicos compartilhados.
2. **Jogador maior** (market maker) e possivel Stackelberg MFG.
3. **Aceleracao numerica** com Newton/policy iteration e controle adaptativo da relaxacao.
4. **Estados adicionais** (preco resiliente, spread) tornando FP multi-dimensional.
5. **Calibracao empirica** com dados reais de order-flow.

## Licenca

MIT.
