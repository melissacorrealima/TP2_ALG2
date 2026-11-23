"""Parser oficial para instâncias TSPLIB95 com métrica EUC_2D.

Segue rigorosamente a especificação de Reinelt (1991):
    d(i, j) = nint(sqrt((x_i - x_j)^2 + (y_i - y_j)^2)) = floor(dist + 0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class TSPInstance:
    name: str
    dimension: int
    edge_weight_type: str
    coords: np.ndarray          # shape (N, 2), float64
    dist_matrix: np.ndarray     # shape (N, N), int64 (EUC_2D nint)
    comment: str = ""
    known_optimum: Optional[int] = None
    source_path: Optional[str] = None


# Valores ótimos publicados pela TSPLIB95 / Universidade de Waterloo (EUC_2D)
KNOWN_OPTIMA_EUC2D = {
    "wi29": 27603,
    "dj38": 6656,
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
    "eil76": 538,
    "pr76": 108159,
}


def compute_euc2d_matrix(coords: np.ndarray) -> np.ndarray:
    """Calcula matriz de distâncias EUC_2D segundo padrão TSPLIB95 (nint = floor(d + 0.5))."""
    coords = np.asarray(coords, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    euclid = np.sqrt(np.sum(diff * diff, axis=-1))
    # Especificação TSPLIB: int(x + 0.5) para valores não negativos
    dist = np.floor(euclid + 0.5).astype(np.int64)
    np.fill_diagonal(dist, 0)
    return dist


def parse_tsp_file(filepath: str | Path, force_euc2d: bool = False) -> TSPInstance:
    """Lê um arquivo .tsp no formato TSPLIB95 e retorna uma TSPInstance."""
    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    header = {}
    coords_list: List[Tuple[int, float, float]] = []
    in_coords = False
    in_display = False
    display_list: List[Tuple[int, float, float]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper.startswith("NODE_COORD_SECTION"):
            in_coords = True
            in_display = False
            continue
        if upper.startswith("DISPLAY_DATA_SECTION"):
            in_coords = False
            in_display = True
            continue
        if upper.startswith("EDGE_WEIGHT_SECTION"):
            in_coords = False
            in_display = False
            continue

        if in_coords or in_display:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    idx = int(float(parts[0]))
                    x = float(parts[1])
                    y = float(parts[2])
                    if in_coords:
                        coords_list.append((idx, x, y))
                    else:
                        display_list.append((idx, x, y))
                    continue
                except ValueError:
                    pass
            in_coords = False
            in_display = False

        if ":" in line:
            k, v = line.split(":", 1)
            header[k.strip().upper()] = v.strip()
        else:
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].upper() in {
                "NAME", "TYPE", "COMMENT", "DIMENSION", "EDGE_WEIGHT_TYPE",
                "EDGE_WEIGHT_FORMAT", "DISPLAY_DATA_TYPE", "KNOWN_OPTIMUM"
            }:
                header[parts[0].upper()] = parts[1].strip()

    name = header.get("NAME", path.stem)
    ewt = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    if ewt != "EUC_2D" and not force_euc2d:
        raise ValueError(
            f"Instância '{name}' possui EDGE_WEIGHT_TYPE='{ewt}', mas este parser exige 'EUC_2D'. "
            "Use force_euc2d=True apenas se desejar explicitamente projetar coordenadas 2D em EUC_2D."
        )

    raw_pts = coords_list if coords_list else display_list
    if not raw_pts:
        raise ValueError(f"Nenhuma coordenada 2D encontrada no arquivo {path}.")

    raw_pts.sort(key=lambda t: t[0])
    coords = np.array([[x, y] for _, x, y in raw_pts], dtype=np.float64)
    dim = int(header.get("DIMENSION", len(coords)))
    if len(coords) != dim:
        raise ValueError(f"Dimensão declarada ({dim}) difere do número de coordenadas ({len(coords)}) em {path}.")

    dist = compute_euc2d_matrix(coords)
    known_opt = None
    if "KNOWN_OPTIMUM" in header:
        try:
            known_opt = int(header["KNOWN_OPTIMUM"])
        except ValueError:
            known_opt = None
    if known_opt is None:
        known_opt = KNOWN_OPTIMA_EUC2D.get(name.lower())

    return TSPInstance(
        name=name,
        dimension=dim,
        edge_weight_type="EUC_2D",
        coords=coords,
        dist_matrix=dist,
        comment=header.get("COMMENT", ""),
        known_optimum=known_opt,
        source_path=str(path),
    )


def write_euc2d_tsp_file(
    filepath: str | Path,
    name: str,
    coords: np.ndarray,
    comment: str = "",
) -> None:
    """Salva coordenadas 2D como um arquivo válido no padrão TSPLIB95 (EUC_2D)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(coords)
    lines = [
        f"NAME : {name}",
        f"COMMENT : {comment or f'EUC_2D TSP instance ({n} nodes)'}",
        "TYPE : TSP",
        f"DIMENSION : {n}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    for i, (x, y) in enumerate(coords, start=1):
        lines.append(f"{i} {x:.6f} {y:.6f}")
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tour_cost(dist_matrix: np.ndarray, tour: List[int]) -> int:
    """Calcula o custo exato de um ciclo Hamiltoniano."""
    n = len(tour)
    if n == 0:
        return 0
    total = 0
    for i in range(n):
        total += int(dist_matrix[tour[i], tour[(i + 1) % n]])
    return total


def validate_tour(tour: List[int], n: int) -> None:
    """Valida se `tour` é uma permutação válida de {0, ..., n-1}."""
    if len(tour) != n:
        raise ValueError(f"Tour inválido: tamanho {len(tour)} != {n}.")
    if set(tour) != set(range(n)):
        raise ValueError("Tour inválido: vértices repetidos ou faltantes.")