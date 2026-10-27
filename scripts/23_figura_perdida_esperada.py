"""Figura: pérdida esperada frente a pérdida realizada (cotas mínima y máxima) por decil de PD.

Lee results/perdida_esperada_deciles_<particion>[_mlp].csv, que genera src/perdida_esperada.py,
y dibuja las dos particiones lado a lado (conjunto de prueba).

Uso (desde la raíz del repo, con el entorno activo):
    python scripts/23_figura_perdida_esperada.py                     # con los MLP (por defecto)
    python scripts/23_figura_perdida_esperada.py --modelo referencia  # logística + Ridge
Salida: figures/perdida_esperada_deciles_mlp.png (o ..._referencia.png)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
PARTICIONES = [("aleatoria", "Partición aleatoria"), ("temporal", "Partición temporal")]
AZUL, GRIS = "#0072B2", "#BBBBBB"


def main(a):
    suf = "_mlp" if a.modelo == "mlp" else ""
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    for ax, (part, titulo) in zip(ejes, PARTICIONES):
        ruta = RAIZ / f"results/perdida_esperada_deciles_{part}{suf}.csv"
        if not ruta.exists():
            comando = f"python run.py perdida-esperada --particion {part}" + (" --modelo mlp" if suf else "")
            raise SystemExit(f"Falta {ruta}. Genéralo con: {comando}")
        g = pd.read_csv(ruta)

        x = np.arange(len(g))
        ancho = 0.38
        esperada = g["perdida_esperada_total"] / 1e6
        minima = g["perdida_realizada_min_total"] / 1e6
        maxima = g["perdida_realizada_max_total"] / 1e6

        ax.bar(x - ancho / 2, esperada, ancho, color=AZUL, label="Pérdida esperada")
        ax.bar(x + ancho / 2, maxima - minima, ancho, bottom=minima, color=GRIS, edgecolor="#555555",
               linewidth=0.8, label="Pérdida realizada (de la cota mínima a la máxima)")

        n = f"{int(g['n'].sum()):,}".replace(",", ".")
        ax.set_title(f"{titulo} (n = {n})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(g["decil_pd"].astype(int))
        ax.set_xlabel("Decil de PD (1 = menor riesgo)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)

    ejes[0].set_ylabel("Millones de dólares")
    ejes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()

    salida = RAIZ / "figures" / f"perdida_esperada_deciles{'_mlp' if suf else '_referencia'}.png"
    salida.parent.mkdir(exist_ok=True)
    fig.savefig(salida, dpi=a.dpi)
    print(f"Figura guardada en {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["mlp", "referencia"], default="mlp")
    ap.add_argument("--dpi", type=int, default=200)
    main(ap.parse_args())