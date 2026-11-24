"""Modelagem Estatística, Regressão de Complexidade, Análise de Sensibilidade e Gráficos.

Gera automaticamente:
1. Modelos de regressão de complexidade empírica (tempo e memória vs N) com R² e RMSE
2. Avaliação quantitativa do impacto e trade-off das 5 estratégias de Warm-Start
3. Análise de Sensibilidade (Instâncias Adversariais):
   - Cálculo de gaps de aproximação em relação ao ótimo (BnB ou TSPLIB publicado)
   - Regressão Linear Múltipla padronizada (coeficientes beta, R², RMSE)
   - Árvore de Decisão interpretável (`DecisionTreeRegressor` + importância de features + regras)
4. Figuras em alta resolução (PDF e PNG) em `results/figures/`
5. Tabelas LaTeX prontas para `\input{}` em `results/tables/`

Uso:
    python3 src/analysis.py
    python3 src/analysis.py --tag smoke
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# --- Curvas teóricas para regressão de complexidade ---
def f_quad(n, a, b):
    return a * (n ** 2) + b

def f_n2logn(n, a, b):
    return a * (n ** 2) * np.log2(np.maximum(n, 2)) + b

def f_cubic(n, a, b):
    return a * (n ** 3) + b

def f_exp(n, a, b):
    return a * (b ** n)


def fit_complexity_models(df_approx: pd.DataFrame, df_sub: pd.DataFrame, df_bnb: pd.DataFrame) -> pd.DataFrame:
    """Ajusta curvas teóricas de tempo e memória vs N e retorna tabela com R² e RMSE."""
    records = []

    def _fit_one(label: str, metric: str, x: np.ndarray, y: np.ndarray, candidates: List[Tuple[str, object, list]]):
        if len(x) < 4 or np.all(y <= 0):
            return
        for cname, func, p0 in candidates:
            try:
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
                yhat = func(x, *popt)
                r2 = float(r2_score(y, yhat))
                rmse = float(np.sqrt(mean_squared_error(y, yhat)))
                records.append({
                    "alvo": label,
                    "metrica": metric,
                    "modelo": cname,
                    "param_a": popt[0],
                    "param_b": popt[1],
                    "R2": round(r2, 4),
                    "RMSE": rmse,
                    "n_pontos": len(x),
                })
            except Exception:
                continue

    # 1. Algoritmos aproximados e heurísticas (Tempo e Memória)
    for alg, cands in [
        ("nearest_neighbor", [("O(N^2)", f_quad, [1e-7, 0.0])]),
        ("convex_hull", [("O(N^2)", f_quad, [1e-7, 0.0]), ("O(N^3)", f_cubic, [1e-9, 0.0])]),
        ("twice_around", [("O(N^2)", f_quad, [1e-7, 0.0]), ("O(N^2 log N)", f_n2logn, [1e-8, 0.0])]),
        ("christofides", [("O(N^2)", f_quad, [1e-7, 0.0]), ("O(N^3)", f_cubic, [1e-9, 0.0])]),
    ]:
        sub = df_approx[df_approx["algorithm"] == alg].sort_values("n")
        x = sub["n"].to_numpy(dtype=float)
        _fit_one(alg, "tempo_s", x, sub["time_s"].to_numpy(dtype=float), cands)
        _fit_one(alg, "memoria_kb", x, sub["peak_memory_kb"].to_numpy(dtype=float), [("O(N^2)", f_quad, [1e-2, 1.0])])

    # 2. Subrotinas críticas: Prim MST e Matching (Blossom)
    for subname, alg_filter, cands in [
        ("mst_prim", "twice_around", [("O(N^2)", f_quad, [1e-7, 0.0])]),
        ("min_weight_matching", "christofides", [("O(N^3)", f_cubic, [1e-9, 0.0]), ("O(N^2)", f_quad, [1e-7, 0.0])]),
    ]:
        sub = df_sub[(df_sub["subroutine"] == subname) & (df_sub["algorithm"] == alg_filter)].sort_values("n")
        x = sub["n"].to_numpy(dtype=float)
        _fit_one(f"sub:{subname}", "tempo_s", x, sub["time_s"].to_numpy(dtype=float), cands)

    # 3. Branch-and-Bound (apenas instâncias concluídas sem timeout)
    for ws in ["cold_start", "convex_hull", "christofides"]:
        sub = df_bnb[(df_bnb["warm_start"] == ws) & (~df_bnb["timed_out"])].copy()
        if len(sub) >= 4:
            sub["bnb_time_s"] = pd.to_numeric(sub["bnb_time_s"], errors="coerce")
            sub = sub.dropna(subset=["bnb_time_s"]).sort_values("n")
            x = sub["n"].to_numpy(dtype=float)
            y = sub["bnb_time_s"].to_numpy(dtype=float)
            _fit_one(f"bnb:{ws}", "tempo_s", x, y, [("O(a * b^N)", f_exp, [1e-4, 1.25])])

    return pd.DataFrame(records)


def compute_optima_map(df_approx: pd.DataFrame, df_bnb: pd.DataFrame) -> Dict[str, int]:
    """Constrói mapa instância -> custo ótimo exato (BnB provado ou TSPLIB publicado)."""
    opt_map: Dict[str, int] = {}
    # Primeiro tenta ótimos conhecidos da TSPLIB
    for _, row in df_approx.iterrows():
        ko = row.get("known_optimum", "NA")
        if str(ko) != "NA" and pd.notna(ko):
            opt_map[row["instance"]] = int(float(ko))
    # Sobrescreve/completa com ótimos provados pelo Branch-and-Bound
    proven = df_bnb[df_bnb["optimal_proven"] == True]
    for inst, grp in proven.groupby("instance"):
        opt_map[inst] = int(pd.to_numeric(grp["optimal_cost"]).min())
    return opt_map


def plot_complexity_curves(df_approx: pd.DataFrame, df_sub: pd.DataFrame, df_bnb: pd.DataFrame) -> None:
    # Figura 1: Tempo empírico das heurísticas e aproximações com curvas ajustadas
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    for alg, label, marker in [
        ("nearest_neighbor", "Vizinho Mais Próximo", "o"),
        ("twice_around", "Twice-Around-the-Tree", "s"),
        ("convex_hull", "Fecho Convexo (Inserção)", "^"),
        ("christofides", "Christofides", "D"),
    ]:
        sub = df_approx[df_approx["algorithm"] == alg].sort_values("n")
        x = sub["n"].to_numpy(dtype=float)
        y = sub["time_s"].to_numpy(dtype=float) * 1000.0  # ms
        ax.scatter(x, y, label=label, marker=marker, s=36)
        if len(x) >= 4:
            fn = f_cubic if alg in {"christofides", "convex_hull"} else f_quad
            try:
                popt, _ = curve_fit(fn, x, y, maxfev=8000)
                xx = np.linspace(x.min(), x.max(), 100)
                ax.plot(xx, fn(xx, *popt), linestyle="--", alpha=0.75)
            except Exception:
                pass
    ax.set_xlabel("Número de vértices (N)")
    ax.set_ylabel("Tempo de execução (ms)")
    ax.set_title("(a) Heurísticas e Aproximações (Ajuste Polinomial)")
    ax.legend()

    # Subrotina Matching vs MST no Christofides
    ax2 = axes[1]
    c_sub = df_sub[df_sub["algorithm"] == "christofides"].pivot_table(
        index="n", columns="subroutine", values="time_s", aggfunc="mean"
    ).sort_index() * 1000.0
    if not c_sub.empty:
        c_sub.plot(kind="bar", stacked=True, ax=ax2, width=0.72)
        ax2.set_xlabel("Número de vértices (N)")
        ax2.set_ylabel("Tempo acumulado por subrotina (ms)")
        ax2.set_title("(b) Decomposição de Tempo no Christofides")
        ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_complexidade_aproximados.pdf")
    fig.savefig(FIG_DIR / "fig1_complexidade_aproximados.png")
    plt.close(fig)

    # Figura 2: Branch-and-Bound (Escala logarítmica vs N e Impacto do Warm-Start)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for ws, label, marker in [
        ("cold_start", "Cold Start", "x"),
        ("nearest_neighbor", "Vizinho Mais Próximo", "o"),
        ("convex_hull", "Fecho Convexo", "^"),
        ("twice_around", "Twice-Around", "s"),
        ("christofides", "Christofides", "D"),
    ]:
        sub = df_bnb[(df_bnb["warm_start"] == ws) & (~df_bnb["timed_out"])].copy()
        sub["bnb_time_s"] = pd.to_numeric(sub["bnb_time_s"], errors="coerce")
        sub = sub.dropna(subset=["bnb_time_s"]).sort_values("n")
        if not sub.empty:
            ax.plot(sub["n"], sub["bnb_time_s"], marker=marker, label=label, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Número de vértices (N)")
    ax.set_ylabel("Tempo do Branch-and-Bound (s, escala log)")
    ax.set_title("(a) Escalabilidade do Branch-and-Bound por Warm-Start")
    ax.legend()

    ax2 = axes[1]
    # Redução relativa de nós expandidos em relação ao Cold Start (nas instâncias concluídas por ambos)
    piv_nodes = df_bnb.pivot_table(index="instance", columns="warm_start", values="nodes_expanded", aggfunc="first")
    if "cold_start" in piv_nodes.columns:
        rel = pd.DataFrame()
        for ws in ["nearest_neighbor", "convex_hull", "twice_around", "christofides"]:
            if ws in piv_nodes.columns:
                rel[ws] = (1.0 - piv_nodes[ws] / piv_nodes["cold_start"].replace(0, np.nan)) * 100.0
        means = rel.mean(skipna=True)
        stds = rel.std(skipna=True).fillna(0.0)
        colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756"]
        ax2.bar(means.index, means.values, yerr=stds.values, capsize=4, color=colors[:len(means)], alpha=0.88)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Redução média de nós expandidos vs Cold Start (%)")
        ax2.set_title("(b) Ganho de Poda na Árvore de Busca")
        ax2.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_bnb_warmstart.pdf")
    fig.savefig(FIG_DIR / "fig2_bnb_warmstart.png")
    plt.close(fig)


def run_sensitivity_analysis(
    df_feat: pd.DataFrame, df_approx: pd.DataFrame, opt_map: Dict[str, int]
) -> Tuple[pd.DataFrame, str]:
    """Análise de sensibilidade: correlaciona features topológicas com o gap de aproximação."""
    rows = []
    for _, r in df_approx.iterrows():
        inst = r["instance"]
        if inst in opt_map and opt_map[inst] > 0:
            gap_pct = 100.0 * (float(r["cost"]) - opt_map[inst]) / opt_map[inst]
            rows.append({
                "instance": inst,
                "n": int(r["n"]),
                "algorithm": r["algorithm"],
                "cost": int(r["cost"]),
                "optimum": opt_map[inst],
                "gap_pct": round(gap_pct, 4),
            })
    df_gaps = pd.DataFrame(rows)
    if df_gaps.empty:
        return df_gaps, "Sem ótimos disponíveis para análise de sensibilidade."

    feature_cols = [
        "aspect_ratio", "convex_hull_ratio", "spatial_entropy",
        "clark_evans_r", "mst_max_degree", "mst_leaf_ratio", "nn_dist_cv",
    ]
    merged = df_gaps.merge(df_feat[["instance"] + feature_cols], on="instance", how="inner")
    merged.to_csv(RESULTS_DIR / "sensitivity_gaps_dataset.csv", index=False)

    report_lines = ["=== ANÁLISE DE SENSIBILIDADE (INSTÂNCIAS ADVERSARIAIS) ===\n"]
    summary_imp = {}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for ax_idx, alg in enumerate(["twice_around", "christofides"]):
        sub = merged[merged["algorithm"] == alg].dropna(subset=feature_cols + ["gap_pct"])
        if len(sub) < 5:
            continue
        X = sub[feature_cols].to_numpy(dtype=float)
        y = sub["gap_pct"].to_numpy(dtype=float)

        # 1. Regressão Linear Múltipla (padronizada para comparar peso das features)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        lr = LinearRegression().fit(Xs, y)
        yhat_lr = lr.predict(Xs)
        r2_lr = r2_score(y, yhat_lr)
        rmse_lr = np.sqrt(mean_squared_error(y, yhat_lr))

        # 2. Árvore de Decisão rasa (profundidade 3 para interpretabilidade científica)
        dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42).fit(X, y)
        yhat_dt = dt.predict(X)
        r2_dt = r2_score(y, yhat_dt)

        report_lines.append(f"--- Algoritmo: {alg} (N_instâncias={len(sub)}) ---")
        report_lines.append(f"Gap médio: {y.mean():.2f}% | Gap máx: {y.max():.2f}% ({sub.iloc[np.argmax(y)]['instance']})")
        report_lines.append(f"Regressão Linear Múltipla: R²={r2_lr:.4f}, RMSE={rmse_lr:.4f}%")
        for fname, coef in sorted(zip(feature_cols, lr.coef_), key=lambda kv: abs(kv[1]), reverse=True):
            corr = np.corrcoef(sub[fname], y)[0, 1]
            report_lines.append(f"  {fname:20s}: beta_pad={coef:+7.3f} | corr_Pearson={corr:+.3f}")
        report_lines.append(f"Árvore de Decisão (max_depth=3): R²={r2_dt:.4f}")
        report_lines.append(export_text(dt, feature_names=feature_cols))

        summary_imp[alg] = dict(zip(feature_cols, dt.feature_importances_))

        # Plot: Coeficientes padronizados da regressão múltipla
        ax = axes[ax_idx]
        order = np.argsort(np.abs(lr.coef_))
        ordered_names = [feature_cols[i] for i in order]
        ordered_coefs = lr.coef_[order]
        colors = ["#E45756" if c > 0 else "#4C78A8" for c in ordered_coefs]
        ax.barh(ordered_names, ordered_coefs, color=colors, alpha=0.88)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coeficiente padronizado β (+ piora gap / - melhora gap)")
        ax.set_title(f"Sensibilidade do Gap — {alg} (R²={r2_lr:.2f})")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_sensibilidade_features.pdf")
    fig.savefig(FIG_DIR / "fig3_sensibilidade_features.png")
    plt.close(fig)

    # Figura 4: Qualidade de aproximação (Gap %) por algoritmo e instância
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    piv_gap = df_gaps.pivot_table(index="instance", columns="algorithm", values="gap_pct", aggfunc="first")
    # Ordena por N
    n_order = df_gaps.groupby("instance")["n"].first().sort_values().index
    cols_show = [c for c in ["nearest_neighbor", "twice_around", "christofides", "convex_hull"] if c in piv_gap.columns]
    piv_gap.loc[n_order, cols_show].plot(kind="bar", ax=ax, width=0.78)
    ax.set_ylabel("Gap em relação ao ótimo (%)")
    ax.set_xlabel("Instância (ordenada por N)")
    ax.set_title("Qualidade das Soluções Heurísticas e Aproximadas (Gap %)")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_gaps_por_instancia.pdf")
    fig.savefig(FIG_DIR / "fig4_gaps_por_instancia.png")
    plt.close(fig)

    txt = "\n".join(report_lines)
    (RESULTS_DIR / "sensitivity_summary.txt").write_text(txt, encoding="utf-8")
    return merged, txt


def export_latex_tables(
    df_reg: pd.DataFrame,
    df_bnb: pd.DataFrame,
    df_gaps: pd.DataFrame,
    audit_bnb_path: Path,
) -> None:
    """Exporta tabelas formatadas em LaTeX (booktabs) prontas para \input{} no artigo SBC."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # Tabela 1: Regressão de complexidade
    if not df_reg.empty:
        t1 = df_reg[["alvo", "metrica", "modelo", "R2", "RMSE", "n_pontos"]].copy()
        t1["RMSE"] = t1["RMSE"].map(lambda v: f"{v:.3e}")
        t1.to_latex(TAB_DIR / "tab_regressao_complexidade.tex", index=False, escape=True)

    # Tabela 2: Impacto do Warm-Start (agregado por estratégia)
    if not df_bnb.empty:
        df_b = df_bnb.copy()
        df_b["bnb_num"] = pd.to_numeric(df_b["bnb_time_s"], errors="coerce")
        agg = df_b.groupby("warm_start").agg(
            resolvidas=("optimal_proven", "sum"),
            timeouts=("timed_out", "sum"),
            ub_medio=("initial_ub", "mean"),
            ws_tempo_ms=("warm_start_time_s", lambda s: s.mean() * 1000.0),
            nos_expandidos_med=("nodes_expanded", "median"),
            nos_podados_med=("nodes_pruned_total", "median"),
            tempo_bnb_med_s=("bnb_num", "median"),
        ).reset_index()
        agg.to_latex(TAB_DIR / "tab_warmstart_resumo.tex", index=False, float_format="%.3f", escape=True)

    # Tabela 3: Gaps de aproximação por algoritmo
    if not df_gaps.empty:
        g_agg = df_gaps.groupby("algorithm").agg(
            gap_medio_pct=("gap_pct", "mean"),
            gap_mediano_pct=("gap_pct", "median"),
            gap_max_pct=("gap_pct", "max"),
            gap_std_pct=("gap_pct", "std"),
        ).reset_index()
        g_agg.to_latex(TAB_DIR / "tab_gaps_aproximacao.tex", index=False, float_format="%.2f", escape=True)

    # Tabela 4: Auditoria (se disponível)
    if audit_bnb_path.exists():
        df_aud = pd.read_csv(audit_bnb_path)
        aud_agg = df_aud.groupby("variant").agg(
            tempo_medio_s=("time_s", "mean"),
            nos_expandidos_med=("nodes_expanded", "median"),
            pico_ram_medio_kb=("peak_memory_kb", "mean"),
        ).reset_index()
        aud_agg.to_latex(TAB_DIR / "tab_auditoria_memoria.tex", index=False, float_format="%.3f", escape=True)


def main():
    parser = argparse.ArgumentParser(description="Análise estatística e geração de figuras/tabelas do TP2.")
    parser.add_argument("--tag", type=str, default="", help="Sufixo dos CSVs de entrada (ex: 'smoke').")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    df_feat = pd.read_csv(RESULTS_DIR / f"features{suffix}.csv")
    df_approx = pd.read_csv(RESULTS_DIR / f"heuristics_approximations{suffix}.csv")
    df_sub = pd.read_csv(RESULTS_DIR / f"subroutine_profiling{suffix}.csv")
    df_bnb = pd.read_csv(RESULTS_DIR / f"bnb_warmstart{suffix}.csv")

    print("[1/4] Ajustando curvas de complexidade empírica (R² e RMSE)...")
    df_reg = fit_complexity_models(df_approx, df_sub, df_bnb)
    df_reg.to_csv(RESULTS_DIR / f"regression_models{suffix}.csv", index=False)
    print(df_reg.to_string(index=False))

    print("\n[2/4] Gerando gráficos de complexidade e impacto do Warm-Start...")
    plot_complexity_curves(df_approx, df_sub, df_bnb)

    print("[3/4] Executando Análise de Sensibilidade (Regressão Múltipla + Árvore de Decisão)...")
    opt_map = compute_optima_map(df_approx, df_bnb)
    df_gaps, sens_txt = run_sensitivity_analysis(df_feat, df_approx, opt_map)
    print(sens_txt)

    print("[4/4] Exportando tabelas LaTeX para results/tables/ ...")
    export_latex_tables(df_reg, df_bnb, df_gaps, RESULTS_DIR / "audit_bnb_strategies.csv")
    print(f"\n[CONCLUÍDO] Figuras salvas em {FIG_DIR} e tabelas em {TAB_DIR}")


if __name__ == "__main__":
    main()