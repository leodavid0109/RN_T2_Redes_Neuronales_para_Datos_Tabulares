"""Figura: distribución de la LGD observada vs. predicha (Ridge) en prueba.

Lee results/lgd_lineal_pred.parquet (lo genera src/lgd_lineal.py) y guarda
figures/fig_lgd_lineal_distribucion.png. No reentrena nada.

Uso:
    python scripts/fig_lgd_lineal.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_parquet("results/lgd_lineal_pred.parquet")
Path("figures").mkdir(exist_ok=True)

bins = np.linspace(0, 1, 41)
q = np.linspace(0.01, 0.99, 99)
parts = ["aleatoria", "temporal"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for j, part in enumerate(parts):
    d = df[(df["particion"] == part) & (df["conjunto"] == "prueba")]
    obs, pred = d["lgd_obs"].to_numpy(), d["lgd_pred"].to_numpy()

    # Fila 1: histogramas superpuestos (proporción de créditos por intervalo)
    ax = axes[0, j]
    w_o = np.ones(len(obs)) / len(obs)
    w_p = np.ones(len(pred)) / len(pred)
    ax.hist(obs, bins=bins, weights=w_o, alpha=0.6, label="Observada")
    ax.hist(pred, bins=bins, weights=w_p, alpha=0.6, label="Predicha (Ridge)")
    ax.set_title(f"Partición {part} (prueba, n = {len(d):,})")
    ax.set_xlabel("LGD")
    ax.set_ylabel("Proporción de créditos")
    ax.legend(loc="upper left")

    # Fila 2: gráfico cuantil-cuantil
    ax = axes[1, j]
    ax.plot(np.quantile(obs, q), np.quantile(pred, q), ".", ms=4,
            label="Cuantiles 1 % a 99 %")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ajuste perfecto (y = x)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cuantiles de la LGD observada")
    ax.set_ylabel("Cuantiles de la LGD predicha")
    ax.set_title(f"Q-Q, partición {part}")
    ax.legend(loc="upper left")

    # Números que respaldan el texto del reporte
    print(f"[{part}] n = {len(d):,}")
    print(f"  Observada: media {obs.mean():.3f} | desv {obs.std():.3f} | "
          f"% con LGD = 1: {np.mean(obs >= 0.9999):.1%}")
    print(f"  Predicha:  media {pred.mean():.3f} | desv {pred.std():.3f} | "
          f"mín {pred.min():.3f} | máx {pred.max():.3f} | "
          f"% con LGD = 1: {np.mean(pred >= 0.9999):.1%}")

fig.tight_layout()
fig.savefig("figures/fig_lgd_lineal_distribucion.png", dpi=200)
print("\nGuardada: figures/fig_lgd_lineal_distribucion.png")