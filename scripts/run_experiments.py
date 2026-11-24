"""Orquestrador principal dos experimentos empíricos do TP.

Funcionalidades críticas de engenharia:
- Executa todos os algoritmos aproximados e heurísticas com profiling de tempo (limpo, sem
  overhead de tracemalloc) + medição isolada de pico de memória por subrotina.
- Executa Branch-and-Bound para TODAS as 5 estratégias de Warm-Start:
    1. cold_start
    2. nearest_neighbor
    3. convex_hull
    4. twice_around
    5. christofides
- Respeita estritamente o limite de tempo (padrão: 1800s = 30min), registrando 'NA' quando
  interrompido sem prova de otimalidade, mas preservando nós expandidos, podados e incumbente.
- Salvamento incremental em CSV com flush imediato + modo `--resume` (retoma exatamente de
  onde parou em caso de queda de energia ou interrupção).
- Suporte a `--workers` para paralelizar execuções longas do Branch-and-Bound quando desejado.

Uso típico:
    # 1. Smoke test completo (~1-2 min) para validar tudo antes do run oficial:
    python3 scripts/run_experiments.py --timeout 5 --repeats 1 --tag smoke

    # 2. Experimento OFICIAL (30 minutos por execução):
    python3 scripts/run_experiments.py --timeout 1800 --repeats 3 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.approximations import christofides, twice_around_the_tree
from src.branch_and_bound import solve_bnb_dfs_inplace
from src.features import extract_instance_features
from src.heuristics import cold_start, convex_hull_insertion, nearest_neighbor
from src.profiling_utils import ProfileCollector, measure_peak_memory_kb
from src.tsplib_parser import TSPInstance, parse_tsp_file, validate_tour

INST_DIR = ROOT / "data" / "instances"
RESULTS_DIR = ROOT / "results"

WARM_START_STRATEGIES = [
    "cold_start",
    "nearest_neighbor",
    "convex_hull",
    "twice_around",
    "christofides",
]


def run_heuristic_clean_and_mem(
    instance: TSPInstance, name: str, repeats: int = 3
) -> Dict[str, object]:
    """Executa heurística/aproximação separando tempo limpo (mediana de `repeats`) e pico de memória."""
    dist = instance.dist_matrix
    coords = instance.coords

    def _call(measure_memory: bool = False):
        col = ProfileCollector()
        if name == "cold_start":
            res = cold_start(dist, collector=col, measure_memory=measure_memory)
        elif name == "nearest_neighbor":
            res = nearest_neighbor(dist, start=0, collector=col, measure_memory=measure_memory)
        elif name == "convex_hull":
            res = convex_hull_insertion(coords, dist, collector=col, measure_memory=measure_memory)
        elif name == "twice_around":
            res = twice_around_the_tree(dist, collector=col, measure_memory=measure_memory)
        elif name == "christofides":
            res = christofides(dist, collector=col, measure_memory=measure_memory)
        else:
            raise ValueError(f"Estratégia desconhecida: {name}")
        return res, col

    # Passadas limpas de tempo (sem overhead do tracemalloc)
    times = []
    sub_times_acc: Dict[str, List[float]] = {}
    last_res = None
    for _ in range(max(1, repeats)):
        res, col = _call(measure_memory=False)
        validate_tour(res.tour, instance.dimension)
        times.append(res.total_time_s)
        for sname, sm in col.subroutines.items():
            sub_times_acc.setdefault(sname, []).append(sm.time_s)
        last_res = res

    # Passada dedicada de memória (com tracemalloc por subrotina)
    _, col_mem = _call(measure_memory=True)
    _, total_peak_kb = measure_peak_memory_kb(_call, False)

    row = {
        "instance": instance.name,
        "n": instance.dimension,
        "algorithm": name,
        "cost": int(last_res.cost),
        "time_s": float(np.median(times)),
        "time_min_s": float(np.min(times)),
        "time_max_s": float(np.max(times)),
        "peak_memory_kb": round(total_peak_kb, 3),
        "known_optimum": instance.known_optimum if instance.known_optimum is not None else "NA",
        "tour": last_res.tour,
    }
    sub_rows = []
    for sname, tlist in sub_times_acc.items():
        sub_rows.append({
            "instance": instance.name,
            "n": instance.dimension,
            "algorithm": name,
            "subroutine": sname,
            "time_s": float(np.median(tlist)),
            "peak_memory_kb": round(col_mem.subroutines[sname].peak_mem_kb, 3) if sname in col_mem.subroutines else 0.0,
        })
    return {"summary": row, "subroutines": sub_rows}


def _bnb_worker_task(args: Tuple[str, str, int, List[int], float, float]) -> Dict[str, object]:
    filepath, ws_name, init_ub, init_tour, ws_time, timeout_s = args
    inst = parse_tsp_file(filepath)

    # Execução principal de tempo limpo (sem tracemalloc)
    res = solve_bnb_dfs_inplace(
        dist_matrix=inst.dist_matrix,
        initial_ub=init_ub,
        initial_tour=init_tour,
        warm_start_name=ws_name,
        warm_start_time_s=ws_time,
        time_limit_s=timeout_s,
    )
    if res.best_tour is not None:
        validate_tour(res.best_tour, inst.dimension)

    # Estimativa de pico de memória do BnB (em janela curta <= 2s para não atrasar timeout de 30min)
    _, peak_kb = measure_peak_memory_kb(
        solve_bnb_dfs_inplace,
        inst.dist_matrix,
        init_ub,
        init_tour,
        ws_name,
        ws_time,
        min(timeout_s, 2.0),
    )

    return {
        "instance": inst.name,
        "n": inst.dimension,
        "warm_start": ws_name,
        "initial_ub": res.initial_ub,
        "warm_start_time_s": round(res.warm_start_time_s, 6),
        "best_cost": res.best_cost,
        "optimal_cost": res.best_cost if res.optimal_proven else "NA",
        "optimal_proven": res.optimal_proven,
        "timed_out": res.timed_out,
        "bnb_time_s": "NA" if res.timed_out else round(res.bnb_time_s, 6),
        "elapsed_until_stop_s": round(res.bnb_time_s, 6),
        "total_time_s": "NA" if res.timed_out else round(res.total_time_s, 6),
        "lap_calls_time_s": round(res.lap_calls_time_s, 6),
        "nodes_expanded": res.nodes_expanded,
        "nodes_pruned_bound": res.nodes_pruned_bound,
        "nodes_pruned_infeasible": res.nodes_pruned_infeasible,
        "nodes_pruned_total": res.nodes_pruned_bound + res.nodes_pruned_infeasible,
        "max_depth": res.max_depth,
        "peak_memory_kb": round(peak_kb, 3),
    }


def append_csv(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()


def load_completed_keys(path: Path, key_cols: Tuple[str, ...]) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add(tuple(r[k] for k in key_cols))
    return done


def main():
    parser = argparse.ArgumentParser(description="Executa bateria experimental completa do TP2.")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Limite de tempo por execução do BnB em segundos (padrão: 1800s = 30min).")
    parser.add_argument("--repeats", type=int, default=3, help="Repetições para mediana de tempo nas heurísticas/aproximações.")
    parser.add_argument("--workers", type=int, default=1, help="Número de processos paralelos para Branch-and-Bound (padrão: 1 = sequencial puro).")
    parser.add_argument("--resume", action="store_true", help="Pula instâncias/estratégias já salvas nos CSVs.")
    parser.add_argument("--tag", type=str, default="", help="Sufixo opcional para arquivos de saída (ex: 'smoke').")
    parser.add_argument("--max-n", type=int, default=999, help="Filtra instâncias com N <= max_n (útil para testes rápidos).")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feat_csv = RESULTS_DIR / f"features{suffix}.csv"
    approx_csv = RESULTS_DIR / f"heuristics_approximations{suffix}.csv"
    sub_csv = RESULTS_DIR / f"subroutine_profiling{suffix}.csv"
    bnb_csv = RESULTS_DIR / f"bnb_warmstart{suffix}.csv"

    if not args.resume:
        for p in [feat_csv, approx_csv, sub_csv, bnb_csv]:
            if p.exists():
                p.unlink()

    manifest_path = INST_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tsp_files = [INST_DIR / m["file"] for m in manifest]
    else:
        tsp_files = sorted(INST_DIR.glob("*.tsp"))

    if not tsp_files:
        raise FileNotFoundError(
            f"Nenhuma instância encontrada em {INST_DIR}. Rode primeiro: python3 scripts/prepare_instances.py"
        )

    instances: List[TSPInstance] = []
    for fp in tsp_files:
        inst = parse_tsp_file(fp)
        if inst.dimension <= args.max_n:
            instances.append(inst)
    instances.sort(key=lambda x: (x.dimension, x.name))

    print(f"=== TP2 DCC207 | {len(instances)} instâncias | Timeout BnB = {args.timeout}s | Workers = {args.workers} ===")

    # 1. Features geométricas/topológicas
    feat_fields = [
        "instance", "n", "aspect_ratio", "convex_hull_ratio", "spatial_entropy",
        "clark_evans_r", "mst_max_degree", "mst_leaf_ratio", "mst_degree_std",
        "mst_edge_cv", "nn_dist_cv", "pairwise_dist_cv", "centroid_dist_cv", "mst_weight",
    ]
    done_feat = load_completed_keys(feat_csv, ("instance",))
    for inst in instances:
        if (inst.name,) not in done_feat:
            fdict = extract_instance_features(inst)
            append_csv(feat_csv, feat_fields, fdict)
    print(f"[FASE 1/3] Features geométricas salvas em {feat_csv}")

    # 2. Heurísticas e Algoritmos Aproximados + Profiling de Subrotinas
    approx_fields = [
        "instance", "n", "algorithm", "cost", "time_s", "time_min_s",
        "time_max_s", "peak_memory_kb", "known_optimum",
    ]
    sub_fields = ["instance", "n", "algorithm", "subroutine", "time_s", "peak_memory_kb"]
    done_approx = load_completed_keys(approx_csv, ("instance", "algorithm"))

    # Cache das soluções iniciais para alimentar o Branch-and-Bound
    warm_starts_cache: Dict[Tuple[str, str], Tuple[int, List[int], float]] = {}

    for inst in instances:
        for alg in WARM_START_STRATEGIES:
            res_bundle = run_heuristic_clean_and_mem(inst, alg, repeats=args.repeats)
            srow = res_bundle["summary"]
            warm_starts_cache[(inst.name, alg)] = (srow["cost"], srow["tour"], srow["time_s"])
            if (inst.name, alg) not in done_approx:
                append_csv(approx_csv, approx_fields, srow)
                for sr in res_bundle["subroutines"]:
                    append_csv(sub_csv, sub_fields, sr)
        print(
            f"  [APPROX] {inst.name:14s} (N={inst.dimension:2d}) | "
            f"Cold={warm_starts_cache[(inst.name, 'cold_start')][0]:6d} "
            f"NN={warm_starts_cache[(inst.name, 'nearest_neighbor')][0]:6d} "
            f"CH={warm_starts_cache[(inst.name, 'convex_hull')][0]:6d} "
            f"TW={warm_starts_cache[(inst.name, 'twice_around')][0]:6d} "
            f"CR={warm_starts_cache[(inst.name, 'christofides')][0]:6d}"
        )
    print(f"[FASE 2/3] Heurísticas e subrotinas salvas em {approx_csv} e {sub_csv}")

    # 3. Branch-and-Bound com as 5 estratégias de Warm-Start
    bnb_fields = [
        "instance", "n", "warm_start", "initial_ub", "warm_start_time_s",
        "best_cost", "optimal_cost", "optimal_proven", "timed_out",
        "bnb_time_s", "elapsed_until_stop_s", "total_time_s", "lap_calls_time_s",
        "nodes_expanded", "nodes_pruned_bound", "nodes_pruned_infeasible",
        "nodes_pruned_total", "max_depth", "peak_memory_kb",
    ]
    done_bnb = load_completed_keys(bnb_csv, ("instance", "warm_start"))

    tasks = []
    for inst in instances:
        for ws in WARM_START_STRATEGIES:
            if (inst.name, ws) in done_bnb:
                continue
            ub, tour, ws_t = warm_starts_cache[(inst.name, ws)]
            tasks.append((inst.source_path, ws, ub, tour, ws_t, args.timeout))

    print(f"[FASE 3/3] Executando {len(tasks)} tarefas de Branch-and-Bound (timeout={args.timeout}s)...")
    t_phase3 = time.perf_counter()

    if args.workers <= 1:
        for i, task in enumerate(tasks, start=1):
            row = _bnb_worker_task(task)
            append_csv(bnb_csv, bnb_fields, row)
            status = "TIMEOUT(NA)" if row["timed_out"] else f"OPT={row['optimal_cost']}"
            print(
                f"  [{i:2d}/{len(tasks)}] {row['instance']:14s} N={row['n']:2d} | {row['warm_start']:16s} | "
                f"{status:12s} | exp={row['nodes_expanded']:7d} | pr={row['nodes_pruned_total']:7d} | "
                f"t={row['elapsed_until_stop_s']:7.3f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_map = {ex.submit(_bnb_worker_task, t): t for t in tasks}
            done_count = 0
            for fut in as_completed(future_map):
                done_count += 1
                row = fut.result()
                append_csv(bnb_csv, bnb_fields, row)
                status = "TIMEOUT(NA)" if row["timed_out"] else f"OPT={row['optimal_cost']}"
                print(
                    f"  [{done_count:2d}/{len(tasks)}] {row['instance']:14s} N={row['n']:2d} | {row['warm_start']:16s} | "
                    f"{status:12s} | exp={row['nodes_expanded']:7d} | pr={row['nodes_pruned_total']:7d} | "
                    f"t={row['elapsed_until_stop_s']:7.3f}s"
                )

    print(f"\n[FINALIZADO] Fase 3 concluída em {time.perf_counter() - t_phase3:.1f}s. Dados salvos em {RESULTS_DIR}")


if __name__ == "__main__":
    main()