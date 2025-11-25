

# **Relatório Técnico Avançado: Estratégias Numéricas e Algoritmos de Aceleração para Convergência em Jogos de Campo Médio na Microestrutura de Mercado**

## **1\. Introdução e Contextualização do Problema**

A modelagem da microestrutura de mercado, especificamente no contexto da B3 (Brasil, Bolsa, Balcão), exige uma representação sofisticada das interações entre agentes de alta frequência, investidores institucionais e a liquidez latente do livro de ofertas. A aplicação de Jogos de Campo Médio (Mean Field Games \- MFG) para problemas de inventário e execução ótima representa a fronteira do conhecimento nesta área, permitindo a transição de modelos de "agente representativo" isolado para sistemas que endogeneizam o impacto de mercado e a distribuição de probabilidade dos inventários dos participantes.1

O desafio técnico apresentado envolve a estagnação da convergência numérica em um pipeline de MFG para controle de inventário unidimensional (1D). O sistema acoplado de equações diferenciais parciais (EDP) — composto por uma equação de Hamilton-Jacobi-Bellman (HJB) retroativa no tempo e uma equação de Fokker-Planck (FP) progressiva no tempo — é resolvido via iteração de ponto fixo (Picard). A persistência de um erro residual na ordem de $4 \\times 10^{-2}$, distante do objetivo de $5 \\times 10^{-4}$, é sintomática de inconsistências estruturais na discretização ou de limitações espectrais do operador de iteração em regimes de forte acoplamento.4

Este relatório analisa exaustivamente as causas fundamentais dessa estagnação, focando na quebra da dualidade discreta (propriedade adjunta), na falta de monotonicidade dos esquemas de diferenças finitas e na ineficiência dos métodos de relaxamento simples. Propõe-se uma reformulação técnica baseada na construção rigorosa de operadores adjuntos discretos, esquemas *upwind* monótonos e a implementação de algoritmos de Aceleração de Anderson e Newton-Raphson para restaurar a convergência quadrática ou superlinear.7

### **1.1 A Natureza do Acoplamento HJB-FP na Microestrutura**

Em problemas de execução ótima, o sistema MFG descreve o equilíbrio de Nash onde cada agente otimiza sua velocidade de negociação considerando a distribuição agregada dos inventários dos outros participantes. O acoplamento é bidirecional:

1. **HJB (Otimização Individual):** O agente resolve um problema de controle estocástico onde o custo de execução depende da densidade média $m(t,q)$ (o termo de campo médio), que afeta o preço ou a liquidez disponível.1  
2. **Fokker-Planck (Dinâmica Populacional):** A evolução da densidade $m(t,q)$ é governada pela estratégia ótima $\\alpha^\*(t,q, \\nabla v)$ derivada da HJB. A densidade é transportada pelo fluxo de ordens dos agentes.4

A estagnação numérica relatada sugere que o "loop" de feedback entre essas duas equações não está se fechando matematicamente no domínio discreto. Diferentemente de sistemas fracamente acoplados, modelos de inventário com restrições rígidas ($q \\in \[0, Q\_{max}\]$) e custos de impacto de mercado geram não-linearidades severas e condições de contorno que, se mal implementadas, destroem a propriedade de contração do mapa de ponto fixo.5

## **2\. Análise Teórica da Estagnação Numérica**

Para diagnosticar o erro de $4 \\times 10^{-2}$, é necessário dissecar a formulação matemática contínua e identificar onde a tradução para o domínio discreto falha. A literatura aponta consistentemente para duas falhas primárias: a violação das condições de Barles-Souganidis na HJB e a inconsistência entre os operadores discretos direto e adjunto.7

### **2.1 O Sistema Contínuo e a Estrutura Adjunta**

O sistema de equações para um horizonte finito $T$ e inventário $q$ em um domínio $\\Omega$ é dado por:

$$\\begin{cases} \-\\partial\_t v \- \\nu \\Delta v \+ H(q, \\nabla v, m) \= 0 & \\text{em } (0,T) \\times \\Omega \\\\ \\partial\_t m \- \\nu \\Delta m \- \\text{div}\\left( m \\cdot D\_p H(q, \\nabla v, m) \\right) \= 0 & \\text{em } (0,T) \\times \\Omega \\\\ m(0, \\cdot) \= m\_0(\\cdot), \\quad v(T, \\cdot) \= G(\\cdot, m(T)) \\end{cases}$$  
Aqui, $H(q, p, m) \= \\sup\_{\\alpha} \\{ \-p \\cdot \\alpha \- L(q, \\alpha, m) \\}$ é o Hamiltoniano. A observação crítica é que o operador diferencial na equação de Fokker-Planck é o **adjunto formal** do operador linearizado da equação HJB.11 No contínuo, isso garante a conservação de certas quantidades e a consistência entre a otimização individual e a evolução da massa.

Entretanto, métodos numéricos ingênuos frequentemente discretizam a HJB e a FP separadamente, utilizando esquemas que são consistentes no limite contínuo ($h \\to 0$), mas que não são adjuntos um do outro no nível discreto para um $h$ fixo. Isso cria um "gap de dualidade". O solver iterativo tenta encontrar um ponto fixo que satisfaça simultaneamente duas dinâmicas discretas ligeiramente incompatíveis. O erro de $4 \\times 10^{-2}$ é, muito provavelmente, a manifestação numérica desse gap estrutural.11

### **2.2 Monotonicidade e Soluções de Viscosidade**

A equação HJB é uma EDP não-linear de primeira ou segunda ordem. Para garantir a convergência para a solução de viscosidade (a solução física correta em presença de *kinks* ou não-suavidades na função valor), o esquema numérico deve satisfazer as propriedades de **monotonicidade, consistência e estabilidade**.7

Esquemas de diferenças finitas centrais (Central Difference) para o termo de deriva de primeira ordem ($p \\cdot \\alpha$) não são monótonos. Eles podem gerar oscilações espúrias e violar o princípio do máximo discreto, especialmente em regiões onde o gradiente da função valor muda abruptamente, como próximo às barreiras de inventário.15 A falta de monotonicidade impede que o solver de HJB estabilize, o que, por sua vez, alimenta a equação FP com um campo de velocidades ruidoso, perpetuando o ciclo de erro.17

A solução mandatória é o uso de esquemas **Upwind** (Montante), onde a direção da discretização espacial depende do sinal da velocidade ótima (controle).16 Isso introduz uma dependência não-linear adicional na matriz do sistema, mas é essencial para a estabilidade.

### **2.3 O Papel Espectral do Acoplamento de Campo Médio**

A convergência do método de Picard depende do raio espectral $\\rho$ do Jacobiano do mapa de ponto fixo. Em problemas de execução ótima, o termo de acoplamento (o custo de impacto $\\kappa m \\cdot \\alpha$) atua como um multiplicador na constante de Lipschitz do mapa.6

Se o impacto de mercado for alto (comum na microestrutura da B3 para grandes ordens), o sistema torna-se "rígido" (stiff). O raio espectral pode exceder a unidade ($\\rho \> 1$) ou ficar muito próximo dela, resultando em divergência ou convergência sublinear extremamente lenta. Métodos de relaxamento simples (mix adaptativo) apenas escalam os autovalores, mas muitas vezes falham em comprimir o espectro de forma eficiente quando há múltiplos modos instáveis.8

---

## **3\. Discretização Rigorosa: O Caminho para a Estabilidade**

A superação da estagnação numérica exige uma reformulação da camada de discretização, priorizando a estrutura algébrica exata sobre a ordem de precisão teórica. A seguir, detalha-se a construção de esquemas de diferenças finitas que preservam a dualidade.

### **3.1 Esquemas Upwind e Construção da Matriz HJB**

Para um grid de inventário uniforme $q\_i \= i \\Delta q$, a discretização do termo de transporte $\\alpha \\partial\_q v$ na HJB deve ser adaptativa.17 Seja $\\alpha\_{i}^n$ o controle ótimo no nó $i$ e tempo $n$. O termo é aproximado como:

$$(\\alpha \\partial\_q v)\_i \\approx \\max(\\alpha\_i^n, 0\) \\frac{v\_{i+1}^n \- v\_{i}^n}{\\Delta q} \+ \\min(\\alpha\_i^n, 0\) \\frac{v\_{i}^n \- v\_{i-1}^n}{\\Delta q}$$  
Isso garante que a informação flua apenas da direção "futura" do fluxo de características, prevenindo oscilações numéricas e garantindo que a matriz do sistema resultante seja uma M-matriz (diagonal dominante, autovalores positivos).19

A equação discretizada no tempo (implícita para estabilidade incondicional) assume a forma algébrica:

$$\\frac{v^n \- v^{n+1}}{\\Delta t} \+ A(\\alpha^n) v^n \= F(m^n)$$

Onde $A(\\alpha^n)$ é uma matriz tridiagonal esparsa cujos coeficientes dependem do controle.7

### **3.2 A Propriedade Adjunta Discreta: $M\_{FP} \= (A\_{HJB})^T$**

Esta é a intervenção técnica mais crítica para corrigir o piso de erro. Em vez de discretizar a equação de Fokker-Planck independentemente usando esquemas conservativos de volumes finitos ou diferenças centradas, deve-se construir a matriz de transição da FP **exatamente** como a transposta da matriz HJB.11

Se o passo de atualização da HJB (retroativo) é resolvido como um sistema linear $(I \- \\Delta t A) v^n \= v^{n+1}$, então o passo da FP (progressivo) deve ser:

$$\\frac{m^{n+1} \- m^n}{\\Delta t} \+ A^T m^{n+1} \= 0 \\implies m^{n+1} \= (I \- \\Delta t A^T)^{-1} m^n$$

(Nota: A formulação exata depende se o esquema temporal é Euler Implícito ou Explícito, mas a relação de transposição espacial é invariante).  
Por que isso resolve a estagnação?  
Ao forçar $M\_{FP} \= A^T$, garante-se que o sistema numérico conserve a massa de probabilidade com precisão de máquina (telescoping sum), desde que a matriz $A$ tenha propriedades de soma de linhas nulas (o que ocorre com condições de contorno de Neumann/Reflexão bem postas).11 Se o usuário estiver usando um solver de FP que discretiza $\\partial\_q (m \\alpha)$ usando um estêncil diferente do usado para $\\alpha \\partial\_q v$, o erro de truncamento $O(\\Delta q)$ difere entre as duas equações. O solver iterativo nunca consegue zerar o resíduo porque os "pontos de equilíbrio" das duas equações discretas não coincidem espacialmente.

### **3.3 Tratamento de Condições de Contorno e Pontos Fantasma**

Nas bordas do inventário ($q=0$ e $q=Q$), a implementação incorreta gera perda de massa. Para a HJB, a condição de contorno de estado (não pode vender em 0, não pode comprar em Q) é equivalente a uma condição de Neumann homogênea no limite de viscosidade, ou simplesmente forçar o esquema *upwind* a olhar "para dentro" do domínio.7

Para garantir a propriedade adjunta, o uso de **pontos fantasma** (ghost points) ou a modificação direta do estêncil na matriz deve ser consistente. Se na HJB a derivada em $q=0$ é aproximada por $(-v\_{0} \+ v\_{1})/\\Delta q$, na FP isso deve corresponder exatamente aos fluxos de entrada e saída nos nós correspondentes. A construção manual da matriz transposta é preferível à implementação de condições de contorno separadas para evitar discrepâncias sutis nos coeficientes.23

#### **Tabela 1: Comparação de Esquemas de Discretização e Impacto na Convergência**

| Característica | Diferenças Centrais | Upwind (Monótono) | Upwind \+ Adjunto Discreto |
| :---- | :---- | :---- | :---- |
| **Precisão Espacial** | $O(\\Delta q^2)$ | $O(\\Delta q)$ | $O(\\Delta q)$ |
| **Estabilidade (HJB)** | Condicional (pode oscilar) | Incondicional (M-Matriz) | Incondicional |
| **Conservação de Massa (FP)** | Não garantida sem fluxo conservativo | Depende da implementação | Exata (Precisão de máquina) |
| **Comportamento do Erro** | Estagnação alta ou oscilação | Convergência lenta | Convergência robusta até tolerância |
| **Recomendação para B3** | Não recomendado | Aceitável | **Mandatório** |

---

## **4\. Algoritmos de Aceleração e Solvers Avançados**

Mesmo com a discretização correta, a convergência linear do método de Picard pode ser proibitivamente lenta ou estagnar devido à rigidez do acoplamento. A seguir, apresentam-se métodos superiores ao "mix adaptativo" mencionado pelo usuário.

### **4.1 Aceleração de Anderson (Anderson Mixing)**

A Aceleração de Anderson (AA) é uma técnica de extrapolação que utiliza o histórico das iterações anteriores para construir uma nova estimativa que minimiza o resíduo linearizado. Diferente de métodos de relaxamento que usam apenas o passo $k-1$, a AA utiliza $m$ passos anteriores, comportando-se efetivamente como um método GMRES (Generalized Minimal Residual) aplicado ao problema não-linear de ponto fixo.26

#### **Mecanismo e Implementação**

Seja $G(x) \= x \- \\Psi(x)$ o resíduo da iteração de ponto fixo. A AA busca uma combinação linear dos iterados anteriores $\\bar{x}\_k \= \\sum\_{j=0}^{m\_k} \\alpha\_j x\_{k-j}$ tal que a norma do resíduo correspondente seja minimizada.

**Algoritmo AA para MFG:**

1. Armazenar os últimos $m$ (tipicamente 5 a 10\) vetores de densidade $m\_k$ e resíduos $G\_k$.  
2. Resolver um problema de mínimos quadrados restrito para encontrar os coeficientes $\\alpha$ que minimizam $\\|\\sum \\alpha\_j G\_{k-j}\\|\_2$, sujeito a $\\sum \\alpha\_j \= 1$.18  
3. Calcular o novo iterado extrapolado.  
4. **Passo Crítico (Projeção):** Como a combinação linear não garante positividade, o resultado deve ser projetado de volta ao simplex de probabilidade (garantir $m \\ge 0$ e $\\int m \= 1$). Algoritmos eficientes de projeção no simplex ou normalização simples devem ser aplicados pós-aceleração para manter a estabilidade da FP.28

A literatura recente demonstra que a AA pode acelerar drasticamente a convergência em problemas de transporte de nêutrons e fluxo de fluidos, que compartilham a estrutura de transporte da FP, frequentemente quebrando ciclos limites onde o Picard simples falha.26

### **4.2 Método de Newton-Raphson Global**

Quando a AA não é suficiente para atingir a tolerância de $5 \\times 10^{-4}$ devido a acoplamentos extremamente fortes, o método de Newton-Raphson é a solução definitiva ("nuclear option"). Ele resolve o sistema acoplado $(v, m)$ simultaneamente, tratando-o como uma raiz $F(Z) \= 0$ onde $Z \= (v, m)^T$.5

Construção do Jacobiano:  
Para inventário 1D, o vetor de incógnitas tem tamanho $2 \\times N\_t \\times N\_q$. O Jacobiano $\\mathcal{J}$ possui estrutura de blocos:  
$$\\mathcal{J} \= \\begin{pmatrix} \\partial\_v \\text{HJB} & \\partial\_m \\text{HJB} \\\\ \\partial\_v \\text{FP} & \\partial\_m \\text{FP} \\end{pmatrix}$$

* $\\partial\_v \\text{HJB}$: É a própria matriz de discretização da HJB.  
* $\\partial\_m \\text{FP}$: É a matriz de transporte da FP ($A^T$).  
* $\\partial\_m \\text{HJB}$: É diagonal e contém as derivadas dos custos de acoplamento ($D\_m L$).  
* $\\partial\_v \\text{FP}$: Este é o termo complexo. Representa como a distribuição de massa muda em resposta a alterações na função valor (via controle ótimo). Como $\\alpha^\*$ depende de $\\nabla v$, este bloco envolve a linearização do fluxo $\\text{div}(m \\alpha(v))$. Pode ser computado via diferenciação automática ou diferenças finitas na dependência do controle.9

A convergência do Newton é quadrática, o que significa que uma vez na bacia de atração, o erro cai de $10^{-2}$ para $10^{-4}$ e $10^{-8}$ em pouquíssimas iterações. Para problemas 1D, o sistema linear resultante é esparso e pode ser resolvido eficientemente com solvers diretos (UMFPACK, Pardiso).7

### **4.3 Métodos de Continuação (Homotopia)**

A não-linearidade dos problemas de microestrutura da B3 pode significar que não existe solução única ou que o solver inicial está fora da bacia de atração do Newton. Métodos de continuação incorporam o problema em uma família parametrizada $P(\\lambda)$, onde $\\lambda \\in $ controla a força do acoplamento ou a viscosidade.11

**Estratégia de Scheduling:**

1. Iniciar com $\\lambda \= 0$ (agentes independentes, sem impacto de mercado). Este problema desacoplado resolve-se facilmente.  
2. Usar a solução de $\\lambda\_k$ como estimativa inicial ("warm start") para $\\lambda\_{k+1} \= \\lambda\_k \+ \\Delta \\lambda$.  
3. Incrementar $\\lambda$ até atingir o problema original ($\\lambda=1$).  
   Esta técnica é particularmente eficaz para navegar pelas bifurcações que ocorrem em MFG com custos de congestão ou impacto de preço quadrático, comuns em execução ótima.12

---

## **5\. Roteiro de Implementação para o Pipeline B3**

Com base na análise acima, recomenda-se o seguinte plano de ação corretiva para destravar a convergência do pipeline.

### **Passo 1: Auditoria da Estrutura Algébrica (Fase de Diagnóstico)**

* **Ação:** Verificar explicitamente no código se a matriz utilizada para o passo de tempo da Fokker-Planck é a transposta da matriz da HJB.  
* **Teste:** Executar o solver com uma distribuição uniforme inicial e verificar a conservação de massa. Se $\\sum m\_{i,n}$ variar mais que $10^{-14}$, a implementação das condições de contorno na FP está incorreta. Deve-se forçar a construção via Matrix\_FP \= Matrix\_HJB.transpose().11  
* **Contexto:** O erro de 4e-2 é característico de "violação de adjunto".

### **Passo 2: Implementação de Upwinding Monótono**

* **Ação:** Substituir quaisquer diferenças centrais no termo de deriva $(\\alpha \\cdot \\nabla v)$ por diferenças *upwind* de primeira ordem.  
* **Detalhe:** O estêncil deve mudar dinamicamente a cada iteração e nó, dependendo se o agente está comprando ($\\alpha \> 0$) ou vendendo ($\\alpha \< 0$). Isso introduz viscosidade numérica ($O(h)$), estabilizando a solução de viscosidade necessária pela teoria.16

### **Passo 3: Integração da Aceleração de Anderson**

* **Ação:** Envelopar o loop de Picard existente com um rotina de Anderson.  
* **Parâmetros:** Utilizar histórico $m=5$. Implementar projeção no simplex após a combinação linear para evitar densidades negativas que colapsariam o cálculo de custos logarítmicos ou de impacto.28  
* **Expectativa:** Isso deve quebrar ciclos limites e reduzir o erro de 4e-2 para a tolerância alvo, explorando a linearidade local do operador de ponto fixo.

### **Passo 4: Newton e Continuação (Caso Persista a Estagnação)**

* **Ação:** Se a aceleração falhar (indicando acoplamento muito forte), implementar o método de Newton. Para mitigar a complexidade de derivar o Jacobiano, usar diferenciação automática (e.g., ForwardDiff.jl em Julia ou JAX em Python) apenas nos blocos locais da matriz.  
* **Ação:** Implementar homotopia no parâmetro de aversão a risco ou impacto de mercado, iniciando de um problema "fácil".11

#### **Tabela 2: Seleção de Solvers Baseada no Regime do Problema**

| Regime de Acoplamento | Sintoma Numérico | Estratégia Recomendada | Complexidade de Implementação |
| :---- | :---- | :---- | :---- |
| **Fraco** (Baixo impacto) | Convergência lenta | Picard com Relaxamento | Baixa |
| **Médio** (B3 Padrão) | Estagnação em resíduo alto | **Picard \+ Aceleração Anderson** | Média |
| **Forte** (Alta Frequência/Crise) | Oscilação ou Divergência | **Newton Global \+ Homotopia** | Alta |

---

## **6\. Conclusão**

A estagnação do solver numérico no pipeline de MFG para microestrutura da B3 não é um artefato aleatório, mas uma consequência determinística de escolhas de discretização e algoritmos de solução. A análise detalhada indica que a correção passa necessariamente pelo restabelecimento da **dualidade discreta** entre as equações HJB e FP, garantindo que o sistema numérico respeite as leis de conservação e otimalidade simultaneamente.

A adoção de esquemas **upwind monótonos** eliminará instabilidades locais na função valor, enquanto a construção da matriz FP como a **transposta exata** da matriz HJB fechará o gap de resíduo que atualmente limita a precisão a $4 \\times 10^{-2}$. Para superar a lentidão inerente ao método de ponto fixo em regimes de alta liquidez e impacto, a **Aceleração de Anderson** apresenta-se como a intervenção de melhor custo-benefício, oferecendo convergência superlinear com modificações mínimas no código existente. Caso a complexidade do modelo aumente, a transição para métodos de Newton-Raphson com estratégias de continuação fornecerá a robustez necessária para simulações de nível industrial.