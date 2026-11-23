"""Suíte de testes automatizados (unitários + integração).

Verifica:
1. Regra oficial de arredondamento EUC_2D da TSPLIB95 (nint = floor(d + 0.5))
2. Validade estrutural de todas as rotas (permutação completa sem vértices repetidos)
3. Corretude exata do Branch-and-Bound contra oráculo exato Held-Karp DP em instâncias diversas
4. Concordância total entre todas as 5 estratégias de Warm-Start no Branch-and-Bound
5. Concordância entre as 3 variantes de auditoria (dfs_inplace, dfs_naive_copy, best_first)
6. Garantias teóricas de aproximação (Twice-Around <= 2*OPT, Christofides <= 1.5*OPT)
7. Tratamento de casos degenerados (pontos colineares no Fecho Convexo)
8. Interrupção graciosa por timeout preservando contadores de nós
9. Extração de features sem NaNs e dentro de intervalos matemáticos válidos
"""

from __future__ import annotations

import numpy as np
import pytest

from src.approximations import christofides, twice_around_the_tree
from src.branch_and_bound import (
    solve_bnb_best_first,
    solve_bnb_dfs_copy,
    solve_bnb_dfs_inplace,
)
from src.features import extract_instance_features
from src.heuristics import cold_start, convex_hull_insertion, nearest_neighbor
from src.tsplib_parser import (
    TSPInstance,
    compute_euc2d_matrix,
    parse_tsp_file,
    tour_cost,
    validate_tour,
    write_euc2d_tsp_file,
)


def held_karp_oracle(dist: np.ndarray) -> int:
    """Programação Dinâmica exata O(N^2 * 2^N) usada como oráculo nos testes."""
    n = len(dist)
    INF = 10**18
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1, 1 << n, 2):
        for u in range(n):
            cur = dp[mask][u]
            if cur == INF:
                continue
            rem = ((1 << n) - 1) ^ mask
            while rem:
                lsb = rem & -rem
                v = lsb.bit_length() - 1
                rem ^= lsb
                nm = mask | lsb
                nc = cur + int(dist[u, v])
                if nc < dp[nm][v]:
                    dp[nm][v] = nc
    full = (1 << n) - 1
    return min(dp[full][u] + int(dist[u, 0]) for u in range(1, n))


def make_random_instance(n: int, seed: int, collinear: bool = False) -> TSPInstance:
    rng = np.random.default_rng(seed)
    if collinear:
        x = rng.uniform(0, 500, size=n)
        coords = np.column_stack([x, 1.5 * x + 10.0])
    else:
        coords = rng.uniform(0, 1000, size=(n, 2))
    dist = compute_euc2d_matrix(coords)
    return TSPInstance(name=f"test_{n}_{seed}", dimension=n, edge_weight_type="EUC_2D", coords=coords, dist_matrix=dist)


def test_euc2d_rounding_and_roundtrip(tmp_path):
    coords = np.array([[0.0, 0.0], [0.0, 1.49], [0.0, 1.50], [3.0, 4.0]])
    f = tmp_path / "mini.tsp"
    write_euc2d_tsp_file(f, "mini", coords)
    inst = parse_tsp_file(f)
    assert inst.dimension == 4
    assert inst.dist_matrix[0, 1] == 1   # floor(1.49 + 0.5) = 1
    assert inst.dist_matrix[0, 2] == 2   # floor(1.50 + 0.5) = 2
    assert inst.dist_matrix[0, 3] == 5   # sqrt(9+16) = 5
    assert np.all(np.diag(inst.dist_matrix) == 0)


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 15, 22])
def test_bnb_matches_held_karp_across_all_5_warmstarts(seed):
    inst = make_random_instance(n=10, seed=seed)
    opt = held_karp_oracle(inst.dist_matrix)

    warm_starts = [
        cold_start(inst.dist_matrix),
        nearest_neighbor(inst.dist_matrix),
        convex_hull_insertion(inst.coords, inst.dist_matrix),
        twice_around_the_tree(inst.dist_matrix),
        christofides(inst.dist_matrix),
    ]
    for ws in warm_starts:
        validate_tour(ws.tour, inst.dimension)
        assert ws.cost == tour_cost(inst.dist_matrix, ws.tour)
        assert ws.cost >= opt

        res = solve_bnb_dfs_inplace(
            inst.dist_matrix,
            initial_ub=ws.cost,
            initial_tour=ws.tour,
            warm_start_name=ws.algorithm,
            warm_start_time_s=ws.total_time_s,
            time_limit_s=10.0,
        )
        assert res.optimal_proven is True
        assert res.timed_out is False
        assert res.best_cost == opt
        validate_tour(res.best_tour, inst.dimension)
        assert tour_cost(inst.dist_matrix, res.best_tour) == opt


def test_audit_variants_agree():
    inst = make_random_instance(n=11, seed=99)
    opt = held_karp_oracle(inst.dist_matrix)
    ws = convex_hull_insertion(inst.coords, inst.dist_matrix)

    r1 = solve_bnb_dfs_inplace(inst.dist_matrix, ws.cost, ws.tour, time_limit_s=10.0)
    r2 = solve_bnb_dfs_copy(inst.dist_matrix, ws.cost, ws.tour, time_limit_s=10.0)
    r3 = solve_bnb_best_first(inst.dist_matrix, ws.cost, ws.tour, time_limit_s=10.0)
    assert r1.best_cost == r2.best_cost == r3.best_cost == opt


@pytest.mark.parametrize("seed", [10, 20, 30, 40])
def test_approximation_theoretical_bounds(seed):
    inst = make_random_instance(n=12, seed=seed)
    opt = held_karp_oracle(inst.dist_matrix)
    tw = twice_around_the_tree(inst.dist_matrix)
    cr = christofides(inst.dist_matrix)

    # Pequena tolerância (+n) devido ao arredondamento inteiro nint da TSPLIB
    assert tw.cost <= 2.0 * opt + inst.dimension
    assert cr.cost <= 1.5 * opt + inst.dimension


def test_collinear_points_handled_gracefully():
    inst = make_random_instance(n=9, seed=5, collinear=True)
    ch = convex_hull_insertion(inst.coords, inst.dist_matrix)
    validate_tour(ch.tour, inst.dimension)
    feats = extract_instance_features(inst)
    for k, v in feats.items():
        if isinstance(v, float):
            assert not np.isnan(v) and not np.isinf(v)


def test_timeout_graceful_and_preserves_counters():
    inst = make_random_instance(n=55, seed=77)
    ws = nearest_neighbor(inst.dist_matrix)
    res = solve_bnb_dfs_inplace(
        inst.dist_matrix,
        initial_ub=ws.cost,
        initial_tour=ws.tour,
        warm_start_name="nearest_neighbor",
        time_limit_s=0.05,
    )
    assert res.timed_out is True
    assert res.optimal_proven is False
    assert res.nodes_expanded >= 1
    assert res.best_cost <= ws.cost


def test_feature_extraction_ranges():
    inst = make_random_instance(n=20, seed=123)
    f = extract_instance_features(inst)
    assert f["aspect_ratio"] >= 1.0
    assert 0.0 < f["convex_hull_ratio"] <= 1.0
    assert 0.0 <= f["spatial_entropy"] <= 1.0001
    assert f["mst_max_degree"] >= 2
    assert f["nn_dist_cv"] >= 0.0