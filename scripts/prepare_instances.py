"""Prepara o conjunto padronizado de 15 instâncias EUC_2D baseadas na TSPLIB95.

Auditoria do conjunto TSPLIB95:
    No repositório clássico da TSPLIB95, instâncias pequenas citadas comumente (ulysses16,
    gr24, fri26, bays29, swiss42, att48) NÃO são do tipo EUC_2D (são GEO, EXPLICIT ou ATT).
    As menores instâncias nativas EUC_2D na TSPLIB95 clássica são `eil51` (N=51),
    `berlin52` (N=52) e `st70` (N=70), complementadas pelas instâncias nacionais EUC_2D no
    mesmo formato TSPLIB (`wi29`, N=29; `dj38`, N=38) da Universidade de Waterloo.

    Para cumprir rigorosamente TODAS as restrições do enunciado:
      - 100% tipo EUC_2D com coordenadas 2D reais (necessárias para Fecho Convexo e features);
      - 15 instâncias no intervalo N in [10, 75];
      - 80% (12 instâncias) na Faixa Principal (N <= 40);
      - 20% (3 instâncias) na Faixa Limite (N in [45, 75]: eil51, berlin52, st70).

    O script baixa as bases reais oficiais (com múltiplos mirrors + descompressão .gz)
    e gera subinstâncias determinísticas (prefixos dos primeiros k vértices de eil51,
    berlin52 e st70) + wi29 e dj38 para a Faixa Principal, mantendo as 3 originais
    completas na Faixa Limite.

Uso:
    python3 scripts/prepare_instances.py
    python3 scripts/prepare_instances.py --offline-synthetic   # Apenas para smoke-test sem internet
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tsplib_parser import parse_tsp_file, write_euc2d_tsp_file

RAW_DIR = ROOT / "data" / "raw_tsplib"
INST_DIR = ROOT / "data" / "instances"

TSPLIB_SOURCES = {
    "eil51": [
        "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp/eil51.tsp",
        "https://raw.githubusercontent.com/mastqe/tsplib/master/eil51.tsp",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/eil51.tsp.gz",
    ],
    "berlin52": [
        "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp/berlin52.tsp",
        "https://raw.githubusercontent.com/mastqe/tsplib/master/berlin52.tsp",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/berlin52.tsp.gz",
    ],
    "st70": [
        "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp/st70.tsp",
        "https://raw.githubusercontent.com/mastqe/tsplib/master/st70.tsp",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/st70.tsp.gz",
    ],
    "wi29": [
        "https://www.math.uwaterloo.ca/tsp/world/wi29.tsp",
    ],
    "dj38": [
        "https://www.math.uwaterloo.ca/tsp/world/dj38.tsp",
    ],
}

# Plano das 15 instâncias (12 na faixa principal N<=40 [80%] + 3 na faixa limite N in [45,75] [20%])
PREFIX_PLAN = [
    ("eil51", 10),
    ("berlin52", 12),
    ("st70", 14),
    ("eil51", 16),
    ("berlin52", 18),
    ("st70", 20),
    ("eil51", 22),
    ("berlin52", 25),
    ("st70", 32),
    ("eil51", 35),
]
# + wi29 (N=29) e dj38 (N=38) -> se não baixarem, usamos berlin52_n29 e st70_n38 automaticamente!


def download_file(name: str, urls: list[str], dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 100:
        print(f"[CACHE] {name} já existe em {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"[DOWNLOAD] Tentando {name} de {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 DCC207-UFMG"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if url.endswith(".gz"):
                data = gzip.decompress(data)
            text = data.decode("utf-8", errors="replace")
            if "NODE_COORD_SECTION" not in text.upper():
                raise ValueError("Conteúdo inválido (sem NODE_COORD_SECTION).")
            dest.write_text(text, encoding="utf-8")
            print(f"[OK] {name} salvo em {dest}")
            return True
        except Exception as e:
            print(f"  [FALHA] {url}: {e}")
    return False


def generate_offline_synthetic_suite() -> None:
    """Gera conjunto sintético EUC_2D diversificado (clusters, anéis, grades, uniformes) apenas para testes offline."""
    print("[MODO OFFLINE] Gerando 15 instâncias sintéticas EUC_2D para validação do pipeline...")
    INST_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("syn_unif_10", 10, "uniform", 101),
        ("syn_clus_12", 12, "clustered", 102),
        ("syn_ring_14", 14, "ring", 103),
        ("syn_grid_16", 16, "grid", 104),
        ("syn_unif_18", 18, "uniform", 105),
        ("syn_clus_20", 20, "clustered", 106),
        ("syn_elon_22", 22, "elongated", 107),
        ("syn_unif_25", 25, "uniform", 108),
        ("syn_clus_29", 29, "clustered", 109),
        ("syn_ring_32", 32, "ring", 110),
        ("syn_unif_35", 35, "uniform", 111),
        ("syn_elon_38", 38, "elongated", 112),
        ("syn_unif_51", 51, "uniform", 113),
        ("syn_clus_52", 52, "clustered", 114),
        ("syn_grid_70", 70, "grid", 115),
    ]
    manifest = []
    for name, n, kind, seed in specs:
        rng = np.random.default_rng(seed)
        if kind == "uniform":
            coords = rng.uniform(0, 1000, size=(n, 2))
        elif kind == "clustered":
            centers = rng.uniform(100, 900, size=(max(2, n // 6), 2))
            idx = rng.integers(0, len(centers), size=n)
            coords = centers[idx] + rng.normal(0, 35, size=(n, 2))
        elif kind == "ring":
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.normal(0, 0.05, size=n)
            radii = 400 + rng.normal(0, 25, size=n)
            coords = np.column_stack([500 + radii * np.cos(angles), 500 + radii * np.sin(angles)])
        elif kind == "elongated":
            coords = np.column_stack([rng.uniform(0, 2500, size=n), rng.uniform(0, 200, size=n)])
        else:  # grid
            side = int(np.ceil(np.sqrt(n)))
            pts = [(i * 100.0, j * 100.0) for i in range(side) for j in range(side)]
            coords = np.array(pts[:n], dtype=np.float64) + rng.normal(0, 5.0, size=(n, 2))
        out = INST_DIR / f"{name}.tsp"
        write_euc2d_tsp_file(out, name, coords, comment=f"Synthetic {kind} EUC_2D test instance")
        manifest.append({"name": name, "n": n, "tier": "main" if n <= 40 else "limit", "file": out.name})
    (INST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[CONCLUÍDO] {len(manifest)} instâncias sintéticas salvas em {INST_DIR}")


def prepare_tsplib_suite() -> None:
    INST_DIR.mkdir(parents=True, exist_ok=True)
    # Limpa arquivos .tsp antigos para garantir conjunto exato de 15 instâncias
    for old in INST_DIR.glob("*.tsp"):
        old.unlink()

    ok_core = {}
    for base_name in ["eil51", "berlin52", "st70"]:
        dest = RAW_DIR / f"{base_name}.tsp"
        if not download_file(base_name, TSPLIB_SOURCES[base_name], dest):
            raise RuntimeError(
                f"Não foi possível baixar {base_name}.tsp de nenhum mirror. "
                "Verifique sua conexão ou coloque o arquivo manualmente em data/raw_tsplib/."
            )
        ok_core[base_name] = parse_tsp_file(dest)

    # Tenta baixar wi29 e dj38 (Waterloo National EUC_2D)
    optional = {}
    for nat_name in ["wi29", "dj38"]:
        dest = RAW_DIR / f"{nat_name}.tsp"
        if download_file(nat_name, TSPLIB_SOURCES[nat_name], dest):
            optional[nat_name] = parse_tsp_file(dest)

    manifest = []

    # 1. Faixa Principal (N <= 40): 10 prefixos determinísticos das bases reais TSPLIB EUC_2D
    for base_name, k in PREFIX_PLAN:
        inst = ok_core[base_name]
        sub_name = f"{base_name}_n{k}"
        out = INST_DIR / f"{sub_name}.tsp"
        write_euc2d_tsp_file(
            out,
            sub_name,
            inst.coords[:k],
            comment=f"First {k} nodes of TSPLIB95 {base_name} (EUC_2D)",
        )
        manifest.append({"name": sub_name, "n": k, "tier": "main", "file": out.name, "origin": f"TSPLIB95:{base_name}"})

    # 2. Completa 12 instâncias na Faixa Principal com wi29 (N=29) e dj38 (N=38) ou fallback de prefixos reais
    for nat_name, fb_base, fb_k in [("wi29", "berlin52", 29), ("dj38", "st70", 38)]:
        if nat_name in optional:
            inst = optional[nat_name]
            out = INST_DIR / f"{nat_name}.tsp"
            write_euc2d_tsp_file(out, nat_name, inst.coords, comment=f"TSPLIB National EUC_2D ({nat_name})")
            manifest.append({"name": nat_name, "n": inst.dimension, "tier": "main", "file": out.name, "origin": "Waterloo-TSPLIB"})
        else:
            inst = ok_core[fb_base]
            sub_name = f"{fb_base}_n{fb_k}"
            out = INST_DIR / f"{sub_name}.tsp"
            write_euc2d_tsp_file(
                out, sub_name, inst.coords[:fb_k], comment=f"First {fb_k} nodes of TSPLIB95 {fb_base} (EUC_2D)"
            )
            manifest.append({"name": sub_name, "n": fb_k, "tier": "main", "file": out.name, "origin": f"TSPLIB95:{fb_base}"})

    # 3. Faixa Limite (N in [45, 75]): 3 instâncias nativas completas da TSPLIB95 (20%)
    for base_name in ["eil51", "berlin52", "st70"]:
        inst = ok_core[base_name]
        out = INST_DIR / f"{base_name}.tsp"
        write_euc2d_tsp_file(out, base_name, inst.coords, comment=inst.comment or f"TSPLIB95 {base_name} (EUC_2D)")
        manifest.append({"name": base_name, "n": inst.dimension, "tier": "limit", "file": out.name, "origin": "TSPLIB95"})

    manifest.sort(key=lambda d: (d["n"], d["name"]))
    (INST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_main = sum(1 for m in manifest if m["tier"] == "main")
    n_limit = sum(1 for m in manifest if m["tier"] == "limit")
    print(f"\n[SUCESSO] Conjunto preparado em {INST_DIR}:")
    print(f"  Total: {len(manifest)} instâncias EUC_2D | Faixa Principal (N<=40): {n_main} ({100*n_main/len(manifest):.0f}%) | Faixa Limite (N in [45,75]): {n_limit} ({100*n_limit/len(manifest):.0f}%)")
    for m in manifest:
        print(f"  - {m['name']:15s} | N={m['n']:2d} | tier={m['tier']:5s} | origem={m['origin']}")


def main():
    parser = argparse.ArgumentParser(description="Prepara as 15 instâncias EUC_2D do TP2.")
    parser.add_argument("--offline-synthetic", action="store_true", help="Gera instâncias sintéticas (apenas p/ teste sem internet).")
    args = parser.parse_args()
    if args.offline_synthetic:
        generate_offline_synthetic_suite()
    else:
        prepare_tsplib_suite()


if __name__ == "__main__":
    main()