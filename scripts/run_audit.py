"""Experimento dedicado de Auditoria de Engenharia.

Responde empiricamente às perguntas de auditoria do enunciado:
1. Gargalos de Memória & GC: Compara DFS com cópia completa de matriz (`dfs_naive_copy`)
   vs DFS refatorado in-place com undo-log (`dfs_inplace`) em tempo e pico de RAM.
2. Estratégia da Árvore: Compara Depth-First Search (`dfs_inplace`) vs Best-First Search
   (`best_first`) em nós expandidos, tempo e pico de memória.
3. Isolamento de Subrotinas no Christofides: Mede o peso percentual do Emparelhamento
   Perfeito Mínimo (Blossom) frente à MST (Prim) e Circuito Euleriano conforme N cresce.

Uso:
    python3 scripts/run_audit.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.approximations import christofides
from src.branch_and_bound import (
    solve_bnb_best_first,
    solve_bnb_dfs_copy,
    solve_bnb_dfs_inplace,
)
from src.heuristics import convex_hull_insertion
from src.profiling_utils import ProfileCollector, measure_peak_memory_kb
from src.tsplib_parser import parse_tsp_file

INST_DIR = ROOT / "data" / "instances"
RESULTS_DIR = ROOT / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = INST_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tsp_files = [INST_DIR / m["file"] for m in manifest]
    else:
        tsp_files = sorted(INST_DIR.glob("*.tsp"))

    instances = sorted((parse_tsp_file(fp) for fp in tsp_files), key=lambda x: (x.dimension, x.name))
    # Seleciona instâncias pequenas/médias (N <= 24) para comparação justa entre estratégias
    audit_insts = [inst for inst in instances if inst.dimension <= 24][:6]

    audit_bnb_path = RESULTS_DIR / "audit_bnb_strategies.csv"
    fields = [
        "instance", "n", "variant", "optimal_proven", "time_s",
        "nodes_expanded", "nodes_pruned_total", "max_depth", "peak_memory_kb",
    ]
    rows = []
    print("=== AUDITORIA 1 & 2: Cópia de Matriz vs In-Place | DFS vs Best-First ===")
    for inst in audit_insts:
        ch = convex_hull_insertion(inst.coords, inst.dist_matrix)
        for variant_name, fn in [
            ("dfs_inplace", solve_bnb_dfs_inplace),
            ("dfs_naive_copy", solve_bnb_dfs_copy),
            ("best_first", solve_bnb_best_first),
        ]:
            # Tempo limpo
            res = fn(
                inst.dist_matrix,
                initial_ub=ch.cost,
                initial_tour=ch.tour,
                warm_start_name="convex_hull",
                warm_start_time_s=ch.total_time_s,
                time_limit_s=15.0,
            )
            # Memória isolada (janela curta <= 3s)
            _, peak_kb = measure_peak_memory_kb(
                fn,
                inst.dist_matrix,
                ch.cost,
                ch.tour,
                "convex_hull",
                ch.total_time_s,
                3.0,
            )
            row = {
                "instance": inst.name,
                "n": inst.dimension,
                "variant": variant_name,
                "optimal_proven": res.optimal_proven,
                "time_s": round(res.bnb_time_s, 6),
                "nodes_expanded": res.nodes_expanded,
                "nodes_pruned_total": res.nodes_pruned_bound + res.nodes_pruned_infeasible,
                "max_depth": res.max_depth,
                "peak_memory_kb": round(peak_kb, 3),
            }
            rows.append(row)
            print(
                f"  {inst.name:14s} N={inst.dimension:2d} | {variant_name:14s} | "
                f"t={row['time_s']:7.4f}s | exp={row['nodes_expanded']:6d} | peak={row['peak_memory_kb']:8.1f} KB"
            )

    with audit_bnb_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Auditoria 3: Breakdown percentual de subrotinas do Christofides em todas as instâncias
    audit_chris_path = RESULTS_DIR / "audit_christofides_breakdown.csv"
    cfields = [
        "instance", "n", "mst_time_s", "odd_subgraph_time_s",
        "matching_time_s", "eulerian_time_s", "total_time_s",
        "matching_pct_time", "matching_peak_kb",
    ]
    crows = []
    print("\n=== AUDITORIA 3: Breakdown de Subrotinas no Christofides ===")
    for inst in instances:
        col_t = ProfileCollector()
        christofides(inst.dist_matrix, collector=col_t, measure_memory=False)
        col_m = ProfileCollector()
        christofides(inst.dist_matrix, collector=col_m, measure_memory=True)

        t_mst = col_t.subroutines["mst_prim"].time_s
        t_odd = col_t.subroutines["odd_subgraph"].time_s
        t_mat = col_t.subroutines["min_weight_matching"].time_s
        t_eul = col_t.subroutines["eulerian_shortcut"].time_s
        t_tot = t_mst + t_odd + t_mat + t_eul
        pct = 100.0 * t_mat / t_tot if t_tot > 0 else 0.0
        crow = {
            "instance": inst.name,
            "n": inst.dimension,
            "mst_time_s": round(t_mst, 6),
            "odd_subgraph_time_s": round(t_odd, 6),
            "matching_time_s": round(t_mat, 6),
            "eulerian_time_s": round(t_eul, 6),
            "total_time_s": round(t_tot, 6),
            "matching_pct_time": round(pct, 2),
            "matching_peak_kb": round(col_m.subroutines["min_weight_matching"].peak_mem_kb, 3),
        }
        crows.append(crow)
        print(f"  {inst.name:14s} N={inst.dimension:2d} | total={t_tot*1000:7.2f}ms | Matching={pct:5.1f}% ({crow['matching_peak_kb']:.1f} KB)")

    with audit_chris_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        w.writeheader()
        w.writerows(crows)

    print(f"\n[CONCLUÍDO] Relatórios de auditoria salvos em {audit_bnb_path} e {audit_chris_path}")


if __name__ == "__main__":
    main()