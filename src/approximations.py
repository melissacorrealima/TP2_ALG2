"""Algoritmos aproximados para o TSP Métrico com profiling por subrotina.

Implementa:
- Algoritmo de Prim nativo em O(N^2) para grafos densos (evita overhead de heap em grafos completos)
- Twice-Around-the-Tree (2-aproximação)
- Christofides (1.5-aproximação), com instrumentação explícita do Emparelhamento
  Perfeito de Peso Mínimo (Blossom / NetworkX) e circuito Euleriano (Hierholzer).
"""

from __future__ import annotations

from typing import List, Tuple
import networkx as nx
import numpy as np

from src.heuristics import SolutionResult
from src.profiling_utils import ProfileCollector, profile_subroutine
from src.tsplib_parser import tour_cost


def prim_mst(dist_matrix: np.ndarray) -> Tuple[List[Tuple[int, int, int]], int]:
    """Algoritmo de Prim denso O(N^2) para grafo completo (sem overhead de heap).

    Retorna:
        edges: lista de arestas (u, v, peso) da MST
        total_weight: peso total da MST
    """
    n = len(dist_matrix)
    dist_list = dist_matrix.tolist()
    in_mst = [False] * n
    min_edge = [10**18] * n
    parent = [-1] * n
    min_edge[0] = 0

    edges: List[Tuple[int, int, int]] = []
    total_weight = 0

    for _ in range(n):
        u = -1
        best = 10**18
        for i in range(n):
            if not in_mst[i] and min_edge[i] < best:
                best = min_edge[i]
                u = i
        in_mst[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u, best))
            total_weight += best
        row = dist_list[u]
        for v in range(n):
            if not in_mst[v]:
                w = row[v]
                if w < min_edge[v]:
                    min_edge[v] = w
                    parent[v] = u

    return edges, total_weight


def twice_around_the_tree(
    dist_matrix: np.ndarray,
    collector: ProfileCollector | None = None,
    measure_memory: bool = False,
) -> SolutionResult:
    """Algoritmo aproximado Twice-Around-the-Tree (fator 2 para TSP métrico)."""
    n = len(dist_matrix)
    col = collector if collector is not None else ProfileCollector()

    with profile_subroutine(col, "mst_prim", measure_memory=measure_memory):
        edges, _ = prim_mst(dist_matrix)

    with profile_subroutine(col, "dfs_preorder_shortcut", measure_memory=measure_memory):
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v, _ in edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = [False] * n
        tour: List[int] = []
        stack = [0]
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                tour.append(u)
                # Ordem reversa preserva ordem canônica da DFS recursiva
                for v in reversed(adj[u]):
                    if not visited[v]:
                        stack.append(v)
        cost = tour_cost(dist_matrix, tour)

    total_time = (
        col.subroutines["mst_prim"].time_s
        + col.subroutines["dfs_preorder_shortcut"].time_s
    )
    peak_mem = max(
        col.subroutines["mst_prim"].peak_mem_kb,
        col.subroutines["dfs_preorder_shortcut"].peak_mem_kb,
    )
    return SolutionResult(
        algorithm="twice_around",
        tour=tour,
        cost=cost,
        total_time_s=total_time,
        peak_memory_kb=peak_mem,
        subroutines=col.to_flat_dict(),
    )


def christofides(
    dist_matrix: np.ndarray,
    collector: ProfileCollector | None = None,
    measure_memory: bool = False,
) -> SolutionResult:
    """Algoritmo aproximado de Christofides (fator 1.5 para TSP métrico).

    Subrotinas instrumentadas individualmente:
        1. mst_prim: Árvore Geradora Mínima O(N^2)
        2. odd_subgraph: Identificação de vértices de grau ímpar e construção do subgrafo
        3. min_weight_matching: Emparelhamento Perfeito de Peso Mínimo (Blossom O(|O|^3))
        4. eulerian_shortcut: Multigrafo Euleriano (Hierholzer) + Shortcut Hamiltoniano
    """
    n = len(dist_matrix)
    col = collector if collector is not None else ProfileCollector()
    dist_list = dist_matrix.tolist()

    with profile_subroutine(col, "mst_prim", measure_memory=measure_memory):
        edges, _ = prim_mst(dist_matrix)

    with profile_subroutine(col, "odd_subgraph", measure_memory=measure_memory):
        deg = [0] * n
        for u, v, _ in edges:
            deg[u] += 1
            deg[v] += 1
        odd_nodes = [i for i in range(n) if deg[i] % 2 == 1]

        G_odd = nx.Graph()
        G_odd.add_nodes_from(odd_nodes)
        m_odd = len(odd_nodes)
        for i in range(m_odd):
            u = odd_nodes[i]
            row_u = dist_list[u]
            for j in range(i + 1, m_odd):
                v = odd_nodes[j]
                G_odd.add_edge(u, v, weight=row_u[v])

    with profile_subroutine(col, "min_weight_matching", measure_memory=measure_memory):
        if m_odd > 0:
            matching = nx.algorithms.matching.min_weight_matching(G_odd, weight="weight")
            if len(matching) * 2 != m_odd:
                raise RuntimeError(
                    f"Emparelhamento incompleto: {len(matching)*2} vértices pareados de {m_odd} ímpares."
                )
        else:
            matching = set()

    with profile_subroutine(col, "eulerian_shortcut", measure_memory=measure_memory):
        # Construção do multigrafo Euleriano via lista de adjacência com remoção eficiente por índice
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v, _ in edges:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in matching:
            adj[u].append(v)
            adj[v].append(u)

        # Algoritmo de Hierholzer iterativo
        stack = [0]
        circuit: List[int] = []
        while stack:
            u = stack[-1]
            if adj[u]:
                v = adj[u].pop()
                adj[v].remove(u)
                stack.append(v)
            else:
                circuit.append(stack.pop())

        # Shortcut: mantém apenas a primeira ocorrência de cada vértice
        seen = [False] * n
        tour: List[int] = []
        for u in circuit:
            if not seen[u]:
                seen[u] = True
                tour.append(u)
        cost = tour_cost(dist_matrix, tour)

    total_time = sum(m.time_s for m in col.subroutines.values())
    peak_mem = max((m.peak_mem_kb for m in col.subroutines.values()), default=0.0)
    return SolutionResult(
        algorithm="christofides",
        tour=tour,
        cost=cost,
        total_time_s=total_time,
        peak_memory_kb=peak_mem,
        subroutines=col.to_flat_dict(),
    )