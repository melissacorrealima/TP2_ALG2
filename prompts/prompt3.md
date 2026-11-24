# Prompt 03 — Extração de Métricas Topológicas e Modelagem de Sensibilidade (Instâncias Adversariais)

## Contexto
Para investigar como a geometria interna das instâncias afeta o *gap* de aproximação do *Twice-Around-the-Tree* e do algoritmo de Christofides em relação ao ótimo exato, precisamos extrair métricas espaciais/topológicas (`src/features.py`) e ajustar modelos estatísticos interpretáveis (`src/analysis.py`).

## Solicitação
1. **Extração de métricas geométricas e topológicas:** implemente o cálculo determinístico de:
   - Razão de aspecto (*aspect ratio*) do *bounding box* dos pontos;
   - Proporção de vértices pertencentes ao fecho convexo ($|H_c|/N$);
   - Entropia espacial de Shannon normalizada em grade adaptativa $k \times k$ ($k = \lceil \sqrt{N} \rceil$);
   - Índice de agrupamento espacial de Clark--Evans ($R$);
   - Estatísticas da Árvore Geradora Mínima (MST): grau máximo (presença de *hubs*) e proporção de folhas;
   - Coeficiente de variação (CV) das distâncias ao vizinho mais próximo (heterogeneidade local).
2. **Modelagem estatística:**
   - Ajuste curvas assintóticas ($O(N^2)$, $O(N^2 \log N)$, $O(N^3)$ e $O(a \cdot b^N)$) via mínimos quadrados não lineares, reportando $R^2$ e RMSE;
   - Construa uma Regressão Linear Múltipla com preditores padronizados ($z$-score) e uma Árvore de Regressão rasa (`max_depth=3`) para identificar quais características ampliam ou reduzem o *gap* percentual de aproximação.