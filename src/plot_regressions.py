
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

# Dados dos experimentos existentes
df_reg = pd.read_csv(RESULTS / "regression_models.csv")
df_approx = pd.read_csv(
    RESULTS / "heuristics_approximations.csv"
)
df_sub = pd.read_csv(
    RESULTS / "subroutine_profiling.csv"
)
df_bnb = pd.read_csv(
    RESULTS / "bnb_warmstart.csv"
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150
})

COLORS = ["#245789", "#B84A42", "#42865B"]

# Modelos utilizados no analysis.py original
MODELS = {
    "O(N^2)": (
        lambda n, a, b: a * n**2 + b,
        r"$aN^2+b$"
    ),
    "O(N^3)": (
        lambda n, a, b: a * n**3 + b,
        r"$aN^3+b$"
    ),
    "O(N^2 log N)": (
        lambda n, a, b: a * n**2 * np.log2(n) + b,
        r"$aN^2\log_2 N+b$"
    ),
    "O(a * b^N)": (
        lambda n, a, b: a * b**n,
        r"$ab^N$"
    )
}


def plot_model(ax, data, target, metric,
               column, title, ylabel,
               scale=1, log=False):

    x = pd.to_numeric(
        data["n"], errors="coerce"
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        data[column], errors="coerce"
    ).to_numpy(dtype=float) * scale

    valid = np.isfinite(x) & np.isfinite(y)

    if log:
        valid &= y > 0

    x, y = x[valid], y[valid]

    ax.set_title(title)
    ax.set_xlabel("Número de vértices (N)")
    ax.set_ylabel(ylabel)

    if len(x) == 0:
        ax.text(
            0.5, 0.5,
            "Sem dados disponíveis",
            transform=ax.transAxes,
            ha="center"
        )
        return

    ax.scatter(
        x, y,
        color="#333333",
        s=27,
        label="Dados experimentais",
        zorder=3
    )

    regressions = df_reg[
        (df_reg["alvo"] == target) &
        (df_reg["metrica"] == metric)
    ]

    xx = np.linspace(x.min(), x.max(), 300)

    for i, (_, row) in enumerate(
        regressions.iterrows()
    ):
        model = row["modelo"]

        if model not in MODELS:
            continue

        func, label = MODELS[model]

        yy = func(
            xx,
            row["param_a"],
            row["param_b"]
        ) * scale

        yy = np.where(
            np.isfinite(yy), yy, np.nan
        )

        if log:
            yy = np.where(yy > 0, yy, np.nan)

        ax.plot(
            xx, yy,
            color=COLORS[i % len(COLORS)],
            linestyle="--",
            linewidth=1.8,
            label=(
                f"{label} — "
                f"$R^2$={row['R2']:.4f}"
            )
        )

    if log:
        ax.set_yscale("log")

    ax.legend()


def save_figure(fig, filename):

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(
            FIGURES / f"{filename}.{ext}",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(fig)

    print(f"Gerado: {filename}")


algorithms = [
    ("nearest_neighbor", "Vizinho Mais Próximo"),
    ("convex_hull", "Fecho Convexo"),
    ("twice_around", "Twice-around-the-tree"),
    ("christofides", "Christofides")
]


# FIGURA 5 — Complexidade temporal
fig, axes = plt.subplots(
    2, 2, figsize=(10, 7)
)

for ax, (alg, title) in zip(
    axes.flat, algorithms
):
    data = df_approx[
        df_approx["algorithm"] == alg
    ]

    plot_model(
        ax, data, alg,
        "tempo_s", "time_s",
        title, "Tempo (ms)",
        scale=1000
    )

save_figure(
    fig, "fig5_regressoes_temporais"
)


# FIGURA 6 — Complexidade espacial
fig, axes = plt.subplots(
    2, 2, figsize=(10, 7)
)

for ax, (alg, title) in zip(
    axes.flat, algorithms
):
    data = df_approx[
        df_approx["algorithm"] == alg
    ]

    plot_model(
        ax, data, alg,
        "memoria_kb", "peak_memory_kb",
        title, "Memória rastreada (KB)"
    )

save_figure(
    fig, "fig6_regressoes_espaciais"
)


# FIGURA 7 — Subrotinas críticas
fig, axes = plt.subplots(
    1, 2, figsize=(10, 3.8)
)

subroutines = [
    (
        "mst_prim",
        "twice_around",
        "Prim — MST"
    ),
    (
        "min_weight_matching",
        "christofides",
        "Matching — Blossom"
    )
]

for ax, (routine, alg, title) in zip(
    axes, subroutines
):
    data = df_sub[
        (df_sub["subroutine"] == routine) &
        (df_sub["algorithm"] == alg)
    ]

    plot_model(
        ax, data,
        f"sub:{routine}",
        "tempo_s", "time_s",
        title, "Tempo (ms)",
        scale=1000
    )

save_figure(
    fig, "fig7_regressoes_subrotinas"
)


# FIGURA 8 — Branch-and-Bound
fig, axes = plt.subplots(
    1, 3, figsize=(13, 4)
)

strategies = [
    ("cold_start", "Cold Start"),
    ("convex_hull", "Fecho Convexo"),
    ("christofides", "Christofides")
]

for ax, (ws, title) in zip(
    axes, strategies
):
    data = df_bnb[
        (df_bnb["warm_start"] == ws) &
        (~df_bnb["timed_out"])
    ]

    plot_model(
        ax, data,
        f"bnb:{ws}",
        "tempo_s", "bnb_time_s",
        title, "Tempo (s)",
        log=True
    )

    ax.axhline(
        1800,
        color="gray",
        linestyle=":",
        label="Timeout (1800 s)"
    )

    ax.legend()

save_figure(
    fig, "fig8_regressoes_bnb"
)

print("\nTodas as figuras foram geradas.")
