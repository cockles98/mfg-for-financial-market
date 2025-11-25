Vou organizar o diagnóstico em camadas, pra ficar claro o que está acontecendo.

---

**1. Qual é o objetivo do projeto**

- Resolver um MFG contínuo (HJB + Fokker‑Planck) para um “market maker agregado”.
- A partir dessa solução, calcular:
  - volatilidade intradiária normalizada do preço,
  - correlação fluxo‑retorno,
  - desvio‑padrão transversal de inventário.
- Ajustar os parâmetros do modelo (`nu`, `phi`, `gamma_T`, `eta0`, `eta1`, forma da densidade inicial, escala do grid, oferta exógena etc.) até que esses três números fiquem próximos dos alvos empíricos extraídos do COTAHIST (`scripts/calibrate_market_metrics.py` + `src/mfg_finance/validation.py`).

Hoje, o “motor numérico” roda, mas o pipeline de calibração não consegue aproximar as métricas do alvo.

---

**2. Sintomas principais que vimos**

1. **Inventário sempre elevado e pouco responsivo**
   - Em praticamente todos os runs recentes, o `inventory_std` (métrica de dispersão transversal) fica estável em torno de **1.6–1.8**, enquanto o alvo é **1.0** (`validation_summary.csv` em vários `cal-...`).
   - Mesmo quando mudamos `nu`, `phi`, `gamma_T`, `drift_strength`, `supply_scale`, o número quase não sai dessa faixa. Isso indica que a tendência centrípeta do modelo (potencial `phi x²`, penalização terminal `gamma_T`) não está forte o suficiente, ou que o FP/HJB não reagiu como esperado às mudanças.

2. **Correlação fluxo‑retorno muito acima do alvo**
   - O target empírico é da ordem de **10⁻⁴**, mas a simulação fica entre **0.05 e 0.25** em praticamente todos os experimentos.
   - A calibração insiste em empurrar `eta1` (sensibilidade do custo de execução ao fluxo) para o teto permitido, e nós mesmos já colocamos caps sucessivos (0.2, 0.05). Quando `eta1` satura, o erro de correlação deixa de diminuir e o loop fica “sem alavancas”.

3. **Volatilidade melhora, mas não resolve**
   - Com várias alterações de `nu`, `drift_strength` e `supply_scale`, a volatilidade normalizada saiu de valores altos (~0.2–0.3) para algo em torno de **0.05–0.11**, mas ainda acima do alvo de ~0.03.
   - A volatilidade é o único alvo que reagiu de fato às mudanças; inventário e correlação estão muito mais “travados”.

4. **Picard e solver convergem, mas para um regime “errado” economicamente**
   - O `solve_mfg_picard` converge com erro final da ordem de 10⁻² e flags de sucesso verdadeiros (`metrics.json`), indicando que o esquema numérico HJB+FP está estável.
   - Ou seja: matematicamente o fixed point é encontrado, mas o equilíbrio resultante não reproduz as estatísticas de mercado desejadas.

5. **Calibração bate nos limites e para**
   - O loop de calibração (`scripts/calibrate_market_metrics.py` + `src/mfg_finance/validation.py`) faz poucas rodadas (3–6) e termina porque:
     - o erro relativo máximo excede o `calibration_rel_guard`, ou
     - os parâmetros chegam em limites (caps em `nu`, `gamma_T`, `eta1`, escala do grid/densidade).
   - As funções de ajuste (`adjust_parameters`) são multiplicativas simples; quando o sistema está num “platô” onde essas direções não reduzem mais o erro, o algoritmo fica preso.

---

**3. Onde estão, de fato, os problemas**

1. **Rigidez estrutural do modelo para inventário e correlação**
   - A dinâmica do inventário é determinada por:
     - o potencial quadrático `phi x²`,
     - a penalização terminal `gamma_T` na HJB,
     - o termo de difusão `nu` no FP,
     - e a regra de controle `alpha_star` que depende de `eta0`, `eta1`.
   - Em prática, essa combinação está produzindo distribuições de inventário que:
     - têm média quase zero (médias que medimos são pequenas), mas
     - mantêm uma largura transversal quase fixa (~1.7), pouco sensível aos parâmetros.
   - Sem um drift forte dependente do estado ou um custo explícito de inventário no meio do horizonte, o FP “aceita” essa largura e o calibrador só consegue empurrar `gamma_T`/`phi` para cima, o que não altera muito a faixa de variação.

2. **Dependência excessiva de `eta1` e do clearing para a correlação**
   - A correlação fluxo‑retorno nasce basicamente da combinação entre:
     - o fluxo médio (controle × densidade),
     - o mecanismo de clearing (`solve_price_clearing` com `price_sensitivity`),
     - e, se houver, o ruído de preço.
   - Como o fluxo agregado do modelo tem um sinal persistente, o calibrador tenta corrigir isso via `eta1`: aumenta o custo quando o fluxo é grande, para forçar menos agressividade. Mas:
     - `eta1` bate no cap,
     - `price_sensitivity` já foi reduzido ao mínimo seguro,
     - e o ruído de preço já foi reduzido/zerado.
   - Resultado: a correlação estabiliza em ~0.1–0.2 e não responde mais às tentativas de ajuste.

3. **Heurística de calibração limitada**
   - `adjust_parameters` usa regras do tipo “multiplicar por (1 + lr * erro_normalizado)”. Isso é simples, mas:
     - Não olha derivadas cruzadas (por ex., como mexer em `nu` afeta inventário e correlação ao mesmo tempo).
     - Usa caps rígidos que frequentemente são atingidos (especialmente em `gamma_T` e `eta1`).
   - Uma vez que os caps são alcançados e o erro ainda é grande, o algoritmo não tem mais direção útil para seguir.

4. **Acoplamento forte entre métricas**
   - Inventário, volatilidade e correlação são alvos que puxam o modelo em direções muitas vezes conflitantes:
     - Aumentar `nu` ajuda a suavizar preço, mas tende a aumentar a dispersão de inventário.
     - Aumentar `gamma_T` ajuda a punir inventários extremos no terminal, mas empurra o controle para comportamentos que podem aumentar a correlação.
   - Sem uma noção clara de trade‑off (por ex., pesos explícitos num funcional de custo global), o calibrador fica tentando reduzir um erro e piora outro.

---

**4. Em resumo, o “problema do projeto” hoje**

- **Numérico:** o solver HJB+FP está estável e convergente, mas o fixed point encontrado tem inventário transversal e correlação bem maiores que o desejado.
- **Estrutural:** a forma do modelo (potencial quadrático, custo de controle só via `eta0`/`eta1`, clearing atual) é rígida demais para produzir, ao mesmo tempo, volatilidade baixa, correlação quase nula e inventário estreito.
- **Algorítmico:** o esquema de calibração heurístico (ajustes multiplicativos com caps) esgota rapidamente as alavancas (`nu`, `phi`, `gamma_T`, `eta1`, escala do grid/densidade) sem conseguir deslocar o sistema para um regime diferente; ficamos presos em uma classe de soluções “parecidas”, com métricas ruins.

Se quiser, posso agora propor um plano mais “cirúrgico” focado em uma dessas camadas (por exemplo, mudar a função de custo no HJB para penalizar inventário diretamente, ou simplificar o clearing e tratar o preço como exógeno) e detalhar as alterações código a código.

