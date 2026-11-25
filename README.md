# Trabalho Prático 2: Análise Empírica e Otimização para o Caixeiro Viajante 

## Integrantes
- Aynoa Souza Jorgino
- Melissa Costa Corrêa de Lima

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

### Suíte Automatizada de Corretude
Valida arredondamento `EUC_2D`, igualdade exata do Branch-and-Bound contra Programação Dinâmica (Held-Karp) nas 5 estratégias de *warm-start*, equivalência entre variantes de auditoria, garantias teóricas ($2\times$ e $1{,}5\times$), pontos colineares, interrupção por *timeout* e extração de *features*[cite: 13]:

```bash
pytest
```

### Preparação do Conjunto Oficial de 15 Instâncias `EUC_2D`
Baixa as bases oficiais `EUC_2D` da TSPLIB95 (`eil51`, `berlin52`, `st70`) e da extensão nacional de Waterloo (`wi29`, `dj38`), gerando 12 instâncias na Faixa Principal ($N \le 40$, 80%) e 3 instâncias nativas completas na Faixa Limite ($N \in [51, 70]$, 20%)[cite: 3, 4, 16]:

```bash
python3 scripts/prepare_instances.py
```

### Auditoria de Engenharia (Memória/GC, DFS vs Best-First, Blossom)
Gera os CSVs oficiais utilizados na Seção 2 do relatório (`results/audit_bnb_strategies.csv` e `results/audit_christofides_breakdown.csv`)[cite: 10, 14]:

```bash
python3 scripts/run_audit.py
```

### Smoke Test Ponta a Ponta da Pipeline Experimental
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

### Geração de Regressões, Gráficos e Tabelas LaTeX
Calcula os modelos de complexidade empírica ($R^2$ e RMSE), executa a Análise de Sensibilidade (Regressão Múltipla padronizada e Árvore de Decisão) e salva os arquivos em `results/figures/` e `results/tables/`:

```bash
python3 src/analysis.py
cat results/sensitivity_summary.txt
```
