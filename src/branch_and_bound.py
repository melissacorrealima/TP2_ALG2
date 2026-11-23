"""Algoritmo Exato Branch-and-Bound para o TSP Simétrico (Relaxação de Atribuição / Carpaneto-Toth).

Arquiteturarefatorada:
1. Lower Bound: Problema de Atribuição Linear (LAP - Hungarian/Jonker-Volgenant em C via SciPy).
   Todo ciclo Hamiltoniano define uma permutação de custo idêntico, logo LAP(M) <= C*.
2. Ramificação (Branching): Partição disjunta sobre o menor subciclo (subtour elimination):
   Dado o menor subciclo C = (e_0, e_1, ..., e_{k-1}), geram-se k subproblemas mutuamente exclusivos:
       - Ramo 0: proíbe e_0
       - Ramo 1: força e_0, proíbe e_1
       - ...
       - Ramo k-1: força e_0, ..., e_{k-2}, proíbe e_{k-1}
3. Refatoração de Memória:
   - Versão principal (`solve_bnb_dfs_inplace`): muta UMA ÚNICA matriz NumPy in-place com
     pilha de restauração (undo-log). Complexidade espacial O(N^2) constante, eliminando
     alocações de matrizes e pausas de Garbage Collection (GC).
   - Versão de auditoria (`solve_bnb_dfs_copy`): copia a matriz `mat.copy()` em cada nó,
     permitindo demonstrar empiricamente o impacto das cópias na memória e no tempo.
   - Versão Best-First (`solve_bnb_best_first`): fila de prioridade (min-heap por LB),
     permitindo comparar empiricamente expansão de nós vs consumo exponencial de RAM.
"""

from __future__ import annotations

import heapq
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tsplib_parser import tour_cost

INF_COST = 1e12


@dataclass
class BnBResult:
    strategy: str
    warm_start_name: str
    initial_ub: int
    warm_start_time_s: float
    best_cost: int
    best_tour: Optional[List[int]]
    optimal_proven: bool
    timed_out: bool
    bnb_time_s: float
    total_time_s: float
    nodes_expanded: int
    nodes_pruned_bound: int
    nodes_pruned_infeasible: int
    max_depth: int
    peak_memory_kb: float = 0.0
    lap_calls_time_s: float = 0.0


def _find_cycles(col_ind: np.ndarray, n: int) -> List[List[int]]:
    visited = [False] * n
    cycles: List[List[int]] = []
    for i in range(n):
        if not visited[i]:
            cyc: List[int] = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cyc.append(curr)
                curr = int(col_ind[curr])
            cycles.append(cyc)
    return cycles


def solve_bnb_dfs_inplace(
    dist_matrix: np.ndarray,
    initial_ub: int,
    initial_tour: Optional[List[int]] = None,
    warm_start_name: str = "custom",
    warm_start_time_s: float = 0.0,
    time_limit_s: float = 1800.0,
) -> BnBResult:
    """Branch-and-Bound DFS refatorado (in-place com undo-log, RAM O(N^2))."""
    n = len(dist_matrix)
    if sys.getrecursionlimit() < 20000:
        sys.setrecursionlimit(20000)

    mat = dist_matrix.astype(np.float64).copy()
    np.fill_diagonal(mat, INF_COST)

    t0 = time.perf_counter()
    best_cost = int(initial_ub)
    best_tour = list(initial_tour) if initial_tour is not None else None
    forced_edges = set()

    nodes_expanded = 0
    nodes_pruned_bound = 0
    nodes_pruned_infeasible = 0
    max_depth = 0
    timed_out = False
    lap_time_acc = 0.0

    def dfs(depth: int) -> None:
        nonlocal best_cost, best_tour, nodes_expanded, nodes_pruned_bound
        nonlocal nodes_pruned_infeasible, max_depth, timed_out, lap_time_acc

        if timed_out:
            return
        if (nodes_expanded & 63) == 0 and (time.perf_counter() - t0) >= time_limit_s:
            timed_out = True
            return
        if depth > max_depth:
            max_depth = depth

        nodes_expanded += 1
        tl0 = time.perf_counter()
        row_ind, col_ind = linear_sum_assignment(mat)
        lap_time_acc += time.perf_counter() - tl0

        lb = float(mat[row_ind, col_ind].sum())
        if lb >= INF_COST * 0.5:
            nodes_pruned_infeasible += 1
            return
        if lb >= best_cost - 1e-9:
            nodes_pruned_bound += 1
            return

        cycles = _find_cycles(col_ind, n)
        if len(cycles) == 1:
            candidate_tour = cycles[0]
            candidate_cost = tour_cost(dist_matrix, candidate_tour)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_tour = candidate_tour
            return

        shortest = min(cycles, key=len)
        k = len(shortest)
        edges = [(shortest[i], shortest[(i + 1) % k]) for i in range(k)]

        saved = []
        for idx, (u, v) in enumerate(edges):
            if (u, v) not in forced_edges:
                old_uv = mat[u, v]
                mat[u, v] = INF_COST
                dfs(depth + 1)
                mat[u, v] = old_uv
                if timed_out:
                    break

            if idx < k - 1:
                if mat[u, v] >= INF_COST:
                    break
                changes = []
                was_forced = (u, v) in forced_edges
                if not was_forced:
                    forced_edges.add((u, v))
                    for j in range(n):
                        if j != v and mat[u, j] < INF_COST:
                            changes.append((u, j, mat[u, j]))
                            mat[u, j] = INF_COST
                    for i in range(n):
                        if i != u and mat[i, v] < INF_COST:
                            changes.append((i, v, mat[i, v]))
                            mat[i, v] = INF_COST
                    if mat[v, u] < INF_COST:
                        changes.append((v, u, mat[v, u]))
                        mat[v, u] = INF_COST
                saved.append(((u, v), was_forced, changes))

        for edge, was_forced, changes in reversed(saved):
            if not was_forced:
                forced_edges.remove(edge)
                for i, j, val in changes:
                    mat[i, j] = val

    dfs(0)
    bnb_time = time.perf_counter() - t0
    return BnBResult(
        strategy="dfs_inplace",
        warm_start_name=warm_start_name,
        initial_ub=int(initial_ub),
        warm_start_time_s=warm_start_time_s,
        best_cost=best_cost,
        best_tour=best_tour,
        optimal_proven=not timed_out,
        timed_out=timed_out,
        bnb_time_s=bnb_time,
        total_time_s=warm_start_time_s + bnb_time,
        nodes_expanded=nodes_expanded,
        nodes_pruned_bound=nodes_pruned_bound,
        nodes_pruned_infeasible=nodes_pruned_infeasible,
        max_depth=max_depth,
        lap_calls_time_s=lap_time_acc,
    )


def solve_bnb_dfs_copy(
    dist_matrix: np.ndarray,
    initial_ub: int,
    initial_tour: Optional[List[int]] = None,
    warm_start_name: str = "custom",
    warm_start_time_s: float = 0.0,
    time_limit_s: float = 60.0,
) -> BnBResult:
    """Versão ingênua (cópia completa de matriz a cada nó) exclusiva para auditoria de memória/GC."""
    n = len(dist_matrix)
    if sys.getrecursionlimit() < 20000:
        sys.setrecursionlimit(20000)

    base = dist_matrix.astype(np.float64).copy()
    np.fill_diagonal(base, INF_COST)

    t0 = time.perf_counter()
    best_cost = int(initial_ub)
    best_tour = list(initial_tour) if initial_tour is not None else None
    nodes_expanded = 0
    nodes_pruned_bound = 0
    nodes_pruned_infeasible = 0
    max_depth = 0
    timed_out = False

    def dfs(mat: np.ndarray, forced: frozenset, depth: int) -> None:
        nonlocal best_cost, best_tour, nodes_expanded, nodes_pruned_bound
        nonlocal nodes_pruned_infeasible, max_depth, timed_out

        if timed_out:
            return
        if (nodes_expanded & 63) == 0 and (time.perf_counter() - t0) >= time_limit_s:
            timed_out = True
            return
        if depth > max_depth:
            max_depth = depth

        nodes_expanded += 1
        row_ind, col_ind = linear_sum_assignment(mat)
        lb = float(mat[row_ind, col_ind].sum())
        if lb >= INF_COST * 0.5:
            nodes_pruned_infeasible += 1
            return
        if lb >= best_cost - 1e-9:
            nodes_pruned_bound += 1
            return

        cycles = _find_cycles(col_ind, n)
        if len(cycles) == 1:
            c = tour_cost(dist_matrix, cycles[0])
            if c < best_cost:
                best_cost = c
                best_tour = cycles[0]
            return

        shortest = min(cycles, key=len)
        k = len(shortest)
        edges = [(shortest[i], shortest[(i + 1) % k]) for i in range(k)]

        cur_mat = mat.copy()
        cur_forced = set(forced)
        for idx, (u, v) in enumerate(edges):
            if (u, v) not in cur_forced:
                child_mat = cur_mat.copy()
                child_mat[u, v] = INF_COST
                dfs(child_mat, frozenset(cur_forced), depth + 1)
                if timed_out:
                    return
            if idx < k - 1:
                if cur_mat[u, v] >= INF_COST:
                    break
                if (u, v) not in cur_forced:
                    cur_forced.add((u, v))
                    cur_mat[u, :] = INF_COST
                    cur_mat[:, v] = INF_COST
                    cur_mat[u, v] = dist_matrix[u, v]
                    cur_mat[v, u] = INF_COST

    dfs(base, frozenset(), 0)
    bnb_time = time.perf_counter() - t0
    return BnBResult(
        strategy="dfs_naive_copy",
        warm_start_name=warm_start_name,
        initial_ub=int(initial_ub),
        warm_start_time_s=warm_start_time_s,
        best_cost=best_cost,
        best_tour=best_tour,
        optimal_proven=not timed_out,
        timed_out=timed_out,
        bnb_time_s=bnb_time,
        total_time_s=warm_start_time_s + bnb_time,
        nodes_expanded=nodes_expanded,
        nodes_pruned_bound=nodes_pruned_bound,
        nodes_pruned_infeasible=nodes_pruned_infeasible,
        max_depth=max_depth,
    )


def solve_bnb_best_first(
    dist_matrix: np.ndarray,
    initial_ub: int,
    initial_tour: Optional[List[int]] = None,
    warm_start_name: str = "custom",
    warm_start_time_s: float = 0.0,
    time_limit_s: float = 60.0,
    max_heap_nodes: int = 50000,
) -> BnBResult:
    """Branch-and-Bound Best-First Search (min-heap por LB) para auditoria comparativa.

    Armazena deltas de arestas proibidas/forçadas para evitar estouro imediato de RAM,
    mas possui trava `max_heap_nodes` para ilustrar o crescimento exponencial da fronteira.
    """
    n = len(dist_matrix)
    base = dist_matrix.astype(np.float64).copy()
    np.fill_diagonal(base, INF_COST)

    t0 = time.perf_counter()
    best_cost = int(initial_ub)
    best_tour = list(initial_tour) if initial_tour is not None else None

    def build_matrix(forbidden: Tuple[Tuple[int, int], ...], forced: Tuple[Tuple[int, int], ...]) -> np.ndarray:
        m = base.copy()
        for u, v in forced:
            val = base[u, v]
            m[u, :] = INF_COST
            m[:, v] = INF_COST
            m[u, v] = val
            m[v, u] = INF_COST
        for u, v in forbidden:
            m[u, v] = INF_COST
        return m

    row_ind, col_ind = linear_sum_assignment(base)
    root_lb = float(base[row_ind, col_ind].sum())

    nodes_expanded = 0
    nodes_pruned_bound = 0
    nodes_pruned_infeasible = 0
    max_depth = 0
    timed_out = False

    # Heap item: (lb, tie_breaker, depth, forbidden_tuple, forced_tuple, col_ind)
    counter = 0
    heap = []
    if root_lb < best_cost - 1e-9:
        heapq.heappush(heap, (root_lb, counter, 0, (), (), col_ind))
    else:
        nodes_pruned_bound += 1

    while heap:
        if (nodes_expanded & 63) == 0 and (time.perf_counter() - t0) >= time_limit_s:
            timed_out = True
            break
        if len(heap) > max_heap_nodes:
            timed_out = True
            break

        lb, _, depth, forbidden, forced, col_ind = heapq.heappop(heap)
        if lb >= best_cost - 1e-9:
            nodes_pruned_bound += 1
            continue

        nodes_expanded += 1
        if depth > max_depth:
            max_depth = depth

        cycles = _find_cycles(col_ind, n)
        if len(cycles) == 1:
            c = tour_cost(dist_matrix, cycles[0])
            if c < best_cost:
                best_cost = c
                best_tour = cycles[0]
            # Em Best-First, o primeiro tour completo extraído é garantidamente ótimo!
            break

        shortest = min(cycles, key=len)
        k = len(shortest)
        edges = [(shortest[i], shortest[(i + 1) % k]) for i in range(k)]

        cur_forced = list(forced)
        cur_forced_set = set(forced)
        forbidden_set = set(forbidden)

        for idx, (u, v) in enumerate(edges):
            if (u, v) not in cur_forced_set:
                child_forb = tuple(sorted(forbidden_set | {(u, v)}))
                child_forc = tuple(sorted(cur_forced_set))
                m = build_matrix(child_forb, child_forc)
                r_i, c_i = linear_sum_assignment(m)
                child_lb = float(m[r_i, c_i].sum())
                if child_lb >= INF_COST * 0.5:
                    nodes_pruned_infeasible += 1
                elif child_lb >= best_cost - 1e-9:
                    nodes_pruned_bound += 1
                else:
                    counter += 1
                    heapq.heappush(heap, (child_lb, counter, depth + 1, child_forb, child_forc, c_i))

            if idx < k - 1:
                if (u, v) in forbidden_set:
                    break
                if (u, v) not in cur_forced_set:
                    cur_forced.append((u, v))
                    cur_forced_set.add((u, v))

    bnb_time = time.perf_counter() - t0
    return BnBResult(
        strategy="best_first",
        warm_start_name=warm_start_name,
        initial_ub=int(initial_ub),
        warm_start_time_s=warm_start_time_s,
        best_cost=best_cost,
        best_tour=best_tour,
        optimal_proven=not timed_out,
        timed_out=timed_out,
        bnb_time_s=bnb_time,
        total_time_s=warm_start_time_s + bnb_time,
        nodes_expanded=nodes_expanded,
        nodes_pruned_bound=nodes_pruned_bound,
        nodes_pruned_infeasible=nodes_pruned_infeasible,
        max_depth=max_depth,
    )