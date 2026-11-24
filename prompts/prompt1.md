# Prompt 01 — Auditoria, Correção de Falhas e Refatoração dos Algoritmos

## Contexto
Implementamos em Python 3 os módulos centrais para o Problema do Caixeiro Viajante Geométrico (`EUC_2D`):
- `src/tsplib_parser.py`: leitura de instâncias TSPLIB95 e cálculo da matriz de distâncias;
- `src/heuristics.py`: estratégias de inicialização *Cold Start*, Vizinho Mais Próximo e Inserção por Fecho Convexo;
- `src/approximations.py`: Árvore Geradora Mínima (Prim), *Twice-Around-the-Tree* e Christofides;
- `src/branch_and_bound.py`: solver exato *Branch-and-Bound* baseado na relaxação do Problema de Atribuição Linear (LAP) com ramificação por eliminação de subciclos.

## Solicitação
Atue como revisor(a) de engenharia de algoritmos e realize uma auditoria técnica completa sobre essas implementações, sem alterar a interface das funções:
1. **Conformidade com a especificação TSPLIB95 (`EUC_2D`):** verifique se o arredondamento das distâncias euclidianas segue rigorosamente o padrão oficial `nint` ($\lfloor d + 0{,}5 \rfloor$) e se há validação estrutural de ciclos Hamiltonianos (permutação completa sem vértices repetidos).
2. **Auditoria de corretude e casos de borda:**
   - No *Branch-and-Bound*, verifique a partição disjunta sobre o menor subciclo para garantir que nenhum ramo válido seja suprimido em subciclos de tamanho 2 (2-ciclos) ou empates de custo;
   - Na Inserção por Fecho Convexo, adicione tratamento seguro (*fallback* via projeção SVD) caso a instância possua pontos colineares que disparem `QhullError`.
3. **Otimização de memória e Garbage Collector:** refatore a recursão do *Branch-and-Bound* para evitar cópias completas da matriz $N \times N$ a cada nó, utilizando mutação *in-place* com pilha de restauração (*undo-log*), mantendo uma variante com cópia e uma variante *Best-First* isoladas apenas para comparação empírica.
4. **Legibilidade e documentação:** adicione *docstrings* e comentários técnicos concisos explicando invariantes, complexidade assintótica e decisões de estrutura de dados em cada subrotina crítica.