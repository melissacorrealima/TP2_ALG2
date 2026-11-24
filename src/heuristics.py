"""Heurísticas construtivas para inicialização (warm-start) do Branch-and-Bound.

Implementa:
1. Cold Start / Trivial: 0 -> 1 -> 2 -> ... -> n-1 -> 0
2. Busca Gulosa (Nearest Neighbor / Vizinho Mais Próximo)
3. Inserção Baseada em Fecho Convexo (Convex Hull Insertion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.profiling_utils import ProfileCollector, profile_subroutine
from src.tsplib_parser import tour_cost


@dataclass
class SolutionResult:
    algorithm: str
    tour: List[int]
    cost: int
    total_time_s: float
    peak_memory_kb: float = 0.0
    subroutines: dict = None

    def __post_init__(self):
        if self.subroutines is None:
            self.subroutines = {}


def cold_start(
    dist_matrix: np.ndarray,
    collector: ProfileCollector | None = None,
    measure_memory: bool = False,
) -> SolutionResult:
    """Estratégia 1: Solução trivial (1 -> 2 -> ... -> n -> 1)."""
    n = len(dist_matrix)
    col = collector if collector is not None else ProfileCollector()
    with profile_subroutine(col, "cold_start", measure_memory=measure_memory):
        tour = list(range(n))
        cost = tour_cost(dist_matrix, tour)
    m = col.subroutines["cold_start"]
    return SolutionResult(
        algorithm="cold_start",
        tour=tour,
        cost=cost,
        total_time_s=m.time_s,
        peak_memory_kb=m.peak_mem_kb,
        subroutines=col.to_flat_dict(),
    )


def nearest_neighbor(
    dist_matrix: np.ndarray,
    start: int = 0,
    collector: ProfileCollector | None = None,
    measure_memory: bool = False,
) -> SolutionResult:
    """Estratégia 2: Heurística Gulosa do Vizinho Mais Próximo (O(N^2))."""
    n = len(dist_matrix)
    col = collector if collector is not None else ProfileCollector()
    with profile_subroutine(col, "nearest_neighbor", measure_memory=measure_memory):
        visited = [False] * n
        visited[start] = True
        tour = [start]
        curr = start
        dist_list = dist_matrix.tolist()
        for _ in range(n - 1):
            row = dist_list[curr]
            best_v = -1
            best_d = 10**18
            for v in range(n):
                if not visited[v] and row[v] < best_d:
                    best_d = row[v]
                    best_v = v
            visited[best_v] = True
            tour.append(best_v)
            curr = best_v
        cost = tour_cost(dist_matrix, tour)

    m = col.subroutines["nearest_neighbor"]
    return SolutionResult(
        algorithm="nearest_neighbor",
        tour=tour,
        cost=cost,
        total_time_s=m.time_s,
        peak_memory_kb=m.peak_mem_kb,
        subroutines=col.to_flat_dict(),
    )


def convex_hull_insertion(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    collector: ProfileCollector | None = None,
    measure_memory: bool = False,
) -> SolutionResult:
    """Estratégia 3: Heurística de Inserção Baseada em Fecho Convexo.

    1. Constrói a sub-rota inicial com os vértices do fecho convexo (ordem anti-horária).
    2. Insere iterativamente o vértice interno k na aresta (i, j) que minimiza
       o custo marginal Delta(i, k, j) = d(i, k) + d(k, j) - d(i, j).
    Trata pontos colineares via fallback geométrico seguro.
    """
    n = len(dist_matrix)
    col = collector if collector is not None else ProfileCollector()
    dist_list = dist_matrix.tolist()

    with profile_subroutine(col, "convex_hull_build", measure_memory=measure_memory):
        if n <= 3:
            tour = list(range(n))
        else:
            try:
                hull = ConvexHull(coords)
                tour = [int(v) for v in hull.vertices]
            except QhullError:
                # Fallback para pontos colineares ou degenerados: ordena pelo eixo principal
                centered = coords - coords.mean(axis=0)
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                proj = centered @ vt[0]
                order = np.argsort(proj)
                tour = [int(order[0]), int(order[-1])]

    with profile_subroutine(col, "convex_hull_insert", measure_memory=measure_memory):
        in_tour = [False] * n
        for v in tour:
            in_tour[v] = True
        remaining = [v for v in range(n) if not in_tour[v]]

        while remaining:
            best_v_idx = -1
            best_pos = -1
            best_delta = 10**18
            m = len(tour)
            for idx_rem, v in enumerate(remaining):
                row_v = dist_list[v]
                for i in range(m):
                    a = tour[i]
                    b = tour[(i + 1) % m]
                    delta = dist_list[a][v] + row_v[b] - dist_list[a][b]
                    if delta < best_delta:
                        best_delta = delta
                        best_v_idx = idx_rem
                        best_pos = i + 1
            chosen_v = remaining.pop(best_v_idx)
            tour.insert(best_pos, chosen_v)

        cost = tour_cost(dist_matrix, tour)

    total_time = (
        col.subroutines["convex_hull_build"].time_s
        + col.subroutines["convex_hull_insert"].time_s
    )
    peak_mem = max(
        col.subroutines["convex_hull_build"].peak_mem_kb,
        col.subroutines["convex_hull_insert"].peak_mem_kb,
    )
    return SolutionResult(
        algorithm="convex_hull",
        tour=tour,
        cost=cost,
        total_time_s=total_time,
        peak_memory_kb=peak_mem,
        subroutines=col.to_flat_dict(),
    )