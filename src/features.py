"""Extração de métricas geométricas e topológicas para Análise de Sensibilidade.

Extrai todas as features sugeridas no enunciado e estatísticas complementares:
- Razão de aspecto (aspect ratio) do bounding box
- Proporção de pontos no fecho convexo
- Entropia espacial (Shannon normalizada em grade adaptativa)
- Índice de clustering espacial de Clark-Evans (R < 1 clusterizado, R > 1 regular)
- Estatísticas da MST: grau máximo (presença de hubs), proporção de folhas, desvio de grau
- Coeficiente de variação (CV) das distâncias ao vizinho mais próximo (heterogeneidade local)
- Coeficiente de variação (CV) das distâncias par-a-par e dispersão radial
"""

from __future__ import annotations

import math
from typing import Dict
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.approximations import prim_mst
from src.tsplib_parser import TSPInstance


def extract_instance_features(instance: TSPInstance) -> Dict[str, float]:
    """Calcula o vetor de características geométricas/topológicas de uma instância TSP EUC_2D."""
    coords = instance.coords
    dist = instance.dist_matrix
    n = instance.dimension

    # 1. Bounding box & Aspect Ratio
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    dx = max(float(xmax - xmin), 1e-9)
    dy = max(float(ymax - ymin), 1e-9)
    aspect_ratio = max(dx, dy) / min(dx, dy)
    area = dx * dy

    # 2. Proporção de pontos pertencentes ao fecho convexo
    try:
        hull = ConvexHull(coords)
        hull_size = len(hull.vertices)
    except QhullError:
        hull_size = min(n, 2)
    convex_hull_ratio = hull_size / float(n)

    # 3. Entropia da distribuição espacial em grade adaptativa k x k (k = ceil(sqrt(N)))
    k = max(2, int(math.ceil(math.sqrt(n))))
    x_norm = np.clip((coords[:, 0] - xmin) / dx, 0.0, 1.0 - 1e-12)
    y_norm = np.clip((coords[:, 1] - ymin) / dy, 0.0, 1.0 - 1e-12)
    gx = (x_norm * k).astype(int)
    gy = (y_norm * k).astype(int)
    cell_ids = gx * k + gy
    _, counts = np.unique(cell_ids, return_counts=True)
    probs = counts / float(n)
    raw_entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(min(n, k * k)) if min(n, k * k) > 1 else 1.0
    spatial_entropy = raw_entropy / max_entropy

    # 4. Distâncias ao vizinho mais próximo (NN) -> CV e índice de Clark-Evans
    dist_f = dist.astype(np.float64).copy()
    np.fill_diagonal(dist_f, np.inf)
    nn_dists = dist_f.min(axis=1)
    nn_mean = float(np.mean(nn_dists))
    nn_std = float(np.std(nn_dists))
    nn_dist_cv = nn_std / nn_mean if nn_mean > 1e-9 else 0.0

    # Clark-Evans R = r_obs / (0.5 / sqrt(N / Area))
    r_expected = 0.5 / math.sqrt(n / max(area, 1e-9))
    clark_evans_r = nn_mean / r_expected if r_expected > 1e-9 else 1.0

    # 5. Estatísticas da MST (presença de hubs, folhas, peso normalizado)
    mst_edges, mst_weight = prim_mst(dist)
    degrees = np.zeros(n, dtype=np.int64)
    mst_edge_weights = []
    for u, v, w in mst_edges:
        degrees[u] += 1
        degrees[v] += 1
        mst_edge_weights.append(w)

    mst_max_degree = int(degrees.max())
    mst_leaf_ratio = float(np.sum(degrees == 1) / float(n))
    mst_degree_std = float(np.std(degrees))
    mst_edge_cv = (
        float(np.std(mst_edge_weights) / np.mean(mst_edge_weights))
        if mst_edge_weights and np.mean(mst_edge_weights) > 1e-9
        else 0.0
    )

    # 6. Estatísticas globais de distâncias par-a-par e dispersão radial
    tri_u = dist[np.triu_indices(n, k=1)].astype(np.float64)
    pw_mean = float(np.mean(tri_u))
    pairwise_dist_cv = float(np.std(tri_u) / pw_mean) if pw_mean > 1e-9 else 0.0

    centroid = coords.mean(axis=0)
    radial = np.linalg.norm(coords - centroid, axis=1)
    radial_mean = float(np.mean(radial))
    centroid_dist_cv = float(np.std(radial) / radial_mean) if radial_mean > 1e-9 else 0.0

    return {
        "instance": instance.name,
        "n": int(n),
        "aspect_ratio": round(aspect_ratio, 6),
        "convex_hull_ratio": round(convex_hull_ratio, 6),
        "spatial_entropy": round(spatial_entropy, 6),
        "clark_evans_r": round(clark_evans_r, 6),
        "mst_max_degree": mst_max_degree,
        "mst_leaf_ratio": round(mst_leaf_ratio, 6),
        "mst_degree_std": round(mst_degree_std, 6),
        "mst_edge_cv": round(mst_edge_cv, 6),
        "nn_dist_cv": round(nn_dist_cv, 6),
        "pairwise_dist_cv": round(pairwise_dist_cv, 6),
        "centroid_dist_cv": round(centroid_dist_cv, 6),
        "mst_weight": int(mst_weight),
    }