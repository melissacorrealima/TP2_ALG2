# Trabalho Prático 2: Análise Empírica e Otimização para o Caixeiro Viajante (TSP)

---

## 1. Configuração do Ambiente (Linux / WSL Ubuntu)

Na raiz do repositório, crie o ambiente virtual e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Roteiro de Testes (Do Mais Rápido ao Mais Completo)

Siga esta ordem para validar o funcionamento antes de iniciar execuções longas:

### Teste 1 — Suíte Automatizada de Corretude (`pytest`) | ~1 segundo
Valida arredondamento `EUC_2D`, igualdade exata do Branch-and-Bound contra Programação Dinâmica (Held-Karp) nas 5 estratégias de *warm-start*, equivalência entre variantes de auditoria, garantias teóricas ($2\times$ e $1{,}5\times$), pontos colineares, interrupção por *timeout* e extração de *features*[cite: 13]:

```bash
pytest
```

### Teste 2 — Verificação Rápida Interativa dos 5 Warm-Starts | ~2 segundos
Executa uma instância sintética de 11 vértices e compara diretamente no terminal o gabarito exato (Held-Karp) com o resultado do Branch-and-Bound para cada inicialização:

```bash
python3 -c "
import numpy as np
from src.tsplib_parser import compute_euc2d_matrix, tour_cost, validate_tour
from src.heuristics import cold_start, nearest_neighbor, convex_hull_insertion
from src.approximations import twice_around_the_tree, christofides
from src.branch_and_bound import solve_bnb_dfs_inplace, solve_bnb_dfs_copy, solve_bnb_best_first

def held_karp(dist):
    n = len(dist)
    INF = 10**18
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1, 1 << n, 2):
        for u in range(n):
            if dp[mask][u] == INF: continue
            rem = ((1 << n) - 1) ^ mask
            while rem:
                lsb = rem & -rem
                v = lsb.bit_length() - 1
                rem ^= lsb
                nc = dp[mask][u] + int(dist[u, v])
                if nc < dp[mask | lsb][v]:
                    dp[mask | lsb][v] = nc
    return min(dp[(1 << n) - 1][u] + int(dist[u, 0]) for u in range(1, n))

coords = np.random.default_rng(42).uniform(0, 500, size=(11, 2))
dist = compute_euc2d_matrix(coords)
opt_real = held_karp(dist)
print(f'Gabarito Exato (Held-Karp): {opt_real}\n')

estrategias = [
    cold_start(dist),
    nearest_neighbor(dist),
    convex_hull_insertion(coords, dist),
    twice_around_the_tree(dist),
    christofides(dist),
]

for res in estrategias:
    validate_tour(res.tour, len(dist))
    assert res.cost == tour_cost(dist, res.tour)
    bnb = solve_bnb_dfs_inplace(dist, res.cost, res.tour, warm_start_name=res.algorithm)
    validate_tour(bnb.best_tour, len(dist))
    assert bnb.best_cost == opt_real
    print(f'{res.algorithm:18s} | inicial={res.cost:4d} | otimo_bnb={bnb.best_cost:4d} | nos_exp={bnb.nodes_expanded:4d} | podados={bnb.nodes_pruned_bound:4d}')

ch = estrategias[2]
assert solve_bnb_dfs_copy(dist, ch.cost, ch.tour).best_cost == solve_bnb_best_first(dist, ch.cost, ch.tour).best_cost == opt_real
print('\nSUCESSO TOTAL: Todos os algoritmos e variantes passaram na validacao!')
"
```

### Teste 3 — Preparação do Conjunto Oficial de 15 Instâncias `EUC_2D` | ~15 segundos
Baixa as bases oficiais `EUC_2D` da TSPLIB95 (`eil51`, `berlin52`, `st70`) e da extensão nacional de Waterloo (`wi29`, `dj38`), gerando 12 instâncias na Faixa Principal ($N \le 40$, 80%) e 3 instâncias nativas completas na Faixa Limite ($N \in [51, 70]$, 20%)[cite: 3, 4, 16]:

```bash
python3 scripts/prepare_instances.py
```

### Teste 4 — Auditoria de Engenharia (Memória/GC, DFS vs Best-First, Blossom) | ~1 minuto
Gera os CSVs oficiais utilizados na Seção 2 do relatório (`results/audit_bnb_strategies.csv` e `results/audit_christofides_breakdown.csv`)[cite: 10, 14]:

```bash
python3 scripts/run_audit.py
```

### Teste 5 — Smoke Test Ponta a Ponta da Pipeline Experimental | ~2 minutos
Valida a integração completa entre coleta experimental (com *timeout* curto de 5 s), regressões estatísticas, geração de gráficos e exportação de tabelas LaTeX:

```bash
python3 scripts/run_experiments.py --timeout 5 --repeats 1 --tag smoke
python3 src/analysis.py --tag smoke
rm -f results/*_smoke*
```

---

## 3. Execução dos Experimentos Oficiais (Timeout = 30 min / 1800 s)

O script `run_experiments.py` salva cada resultado incrementalmente em CSV com `flush()` imediato e suporta `--resume` (caso o processo seja interrompido, basta rodar o comando novamente para continuar exatamente de onde parou)[cite: 15].

### Execução Recomendada em Degraus Crescentes
Rodar a Faixa Principal ($N \le 40$) sequencialmente (`--workers 1`) garante medições de tempo livres de contenção de CPU para o ajuste de curvas assintóticas, enquanto `--workers 4` acelera os testes de estresse da Faixa Limite ($N \in \{51, 52, 70\}$)[cite: 15]:

```bash
# Degrau 1: Instâncias pequenas (N <= 25 -> 40/75 execuções de BnB) | ~1 a 3 min
python3 scripts/run_experiments.py --timeout 1800 --repeats 3 --max-n 25 --workers 1 --resume

# Degrau 2: Instâncias médias (N <= 32 -> 50/75 execuções de BnB) | ~3 a 15 min
python3 scripts/run_experiments.py --timeout 1800 --repeats 3 --max-n 32 --workers 1 --resume

# Degrau 3: Topo da Faixa Principal (N <= 40 -> 60/75 execuções de BnB) | ~45 min a 2h30
python3 scripts/run_experiments.py --timeout 1800 --repeats 3 --max-n 40 --workers 1 --resume

# Degrau 4: Faixa Limite / Testes de Estresse (N = 51, 52, 70 -> 75/75 execuções) | ~2h em 4 núcleos
python3 scripts/run_experiments.py --timeout 1800 --repeats 3 --workers 4 --resume
```

---

## 4. Validação dos CSVs, Modelagem Estatística e Compilação do Relatório

### Verificação de Integridade dos Dados Coletados
Confirma se todas as 75 execuções do Branch-and-Bound foram registradas e se todos os ótimos provados concordam entre as 5 estratégias de *warm-start*:

```bash
python3 -c "
import pandas as pd
b = pd.read_csv('results/bnb_warmstart.csv')
print('Total de execucoes BnB (esperado 75):', len(b))
print('Timeouts por estrategia:\n', b.groupby('warm_start')['timed_out'].sum())
ok = b[b['optimal_proven'] == True].groupby('instance')['best_cost'].nunique()
assert (ok == 1).all(), f'Divergencia de otimo:\n{ok[ok > 1]}'
print('SUCESSO: Todos os otimos provados concordam 100% entre as estrategias!')
"
```

### Geração de Regressões, Gráficos e Tabelas LaTeX
Calcula os modelos de complexidade empírica ($R^2$ e RMSE), executa a Análise de Sensibilidade (Regressão Múltipla padronizada e Árvore de Decisão) e salva os arquivos em `results/figures/` e `results/tables/`:

```bash
python3 src/analysis.py
cat results/sensitivity_summary.txt
```
