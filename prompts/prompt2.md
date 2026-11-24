# Prompt 02 — Suíte de Testes, Instrumentação Experimental e Especificação de Figuras e Tabelas

## Contexto
É necessário estruturar a camada de validação (`tests/test_tsp.py`), o orquestrador experimental (`scripts/run_audit.py` e `scripts/run_experiments.py`) e o gerador de saídas estatísticas (`src/analysis.py`) para fundamentar a redação do relatório científico.

## Requisitos do Enunciado a Atender
- **Conjunto de teste (`EUC_2D`):** 15 instâncias no intervalo $N \in [10, 75]$, sendo 80% (12 instâncias) na Faixa Principal ($N \le 40$) e 20% (3 instâncias nativas: `eil51`, `berlin52`, `st70`) na Faixa Limite ($N \in [45, 75]$).
- **Limite de execução e resiliência:** teto rígido de 30 minutos (1800 s) por execução do *Branch-and-Bound*, registrando estouros como `NA` sem descartar nós expandidos, nós podados e melhor solução incumbente até a interrupção.
- **Profiling em nível de subrotina:** medição separada de tempo (`time.perf_counter()`, mediana de 3 execuções limpas) e pico de memória (`tracemalloc` isolado para não contaminar os tempos com overhead de rastreamento).

## Solicitação
1. **Suíte de testes (`pytest`):** crie testes unitários e de integração validando o arredondamento `EUC_2D`, a igualdade exata do *Branch-and-Bound* contra um oráculo exato de Programação Dinâmica (Held-Karp) nas 5 estratégias de *warm-start*, as cotas teóricas de aproximação ($2\times$ e $1{,}5\times$), pontos colineares e o encerramento gracioso por *timeout*.
2. **Especificação rigorosa de figuras e tabelas (base para o artigo):**
   - **Figuras (`.png` e `.pdf` em alta resolução):**
     - `fig1`: curvas empíricas de tempo das heurísticas/aproximações com ajustes polinomiais sobrepostos + gráfico de barras empilhadas decompondo as subrotinas do Christofides (destacando o peso do *Matching*);
     - `fig2`: escalabilidade do *Branch-and-Bound* em escala logarítmica até o ponto de colapso + barras de redução percentual média de nós expandidos (com desvio padrão) em relação ao *Cold Start*;
     - `fig3`: gráfico horizontal dos coeficientes padronizados ($\beta$) da regressão linear múltipla relacionando métricas topológicas ao *gap* de aproximação;
     - `fig4`: barras comparativas dos *gaps* percentuais em relação ao ótimo ao longo de todas as instâncias ordenadas por $N$.
   - **Tabelas LaTeX (`booktabs`):** gere automaticamente tabelas consolidadas de auditoria de memória/estratégia, regressões de complexidade ($R^2$ e RMSE), resumo agregado do *warm-start* (resolvidas, timeouts, nós e tempos) e estatísticas dos *gaps* de aproximação.