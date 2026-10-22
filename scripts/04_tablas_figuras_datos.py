"""Genera las tablas y figuras del apartado de datos (se regeneran desde el código).

Salidas:  results/tabla01..tabla04 (csv)  y  figures/fig01..fig02 (png).
Uso:  python scripts/04_tablas_figuras_datos.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.objetivo import MAPA_PD, codificar_pd  # noqa: E402

RES, FIG = RAIZ / "results", RAIZ / "figures"
FIG.mkdir(exist_ok=True)

# Paleta (Okabe-Ito): buenos azul, malos naranja; gris = excluidos (NA), a propósito neutro
AZUL, NARANJA, GRIS = "#0072B2", "#D55E00", "#8A8A8A"
TINTA, TINTA_SUAVE = "#222222", "#666666"

df = pd.read_csv(RAIZ / "data" / "raw" / "loan.csv", usecols=["loan_status", "issue_d"], low_memory=False)
df["objetivo"] = codificar_pd(df["loan_status"])
df["anio"] = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year
etiq = df[df["objetivo"].notna()]

# ---- Tabla 1: loan_status y su codificación ----
t1 = df["loan_status"].value_counts().rename("registros").to_frame()
t1["codificacion"] = [("NA (se excluye)" if MAPA_PD[s] is None else str(MAPA_PD[s])) for s in t1.index]
t1["porcentaje"] = (100 * t1["registros"] / len(df)).round(2)
t1.index.name = "loan_status"
t1.to_csv(RES / "tabla01_loan_status.csv")

# ---- Tabla 2: registros por año de originación ----
g = df.groupby("anio")
t2 = pd.DataFrame({
    "total": g.size(),
    "con_etiqueta": g["objetivo"].count(),
    "malos": g["objetivo"].sum().astype(int),
})
t2["excluidos_NA"] = t2["total"] - t2["con_etiqueta"]
t2["excluidos_current"] = df.assign(c=df["loan_status"] == "Current").groupby("anio")["c"].sum().astype(int)
t2["pct_excluidos_que_son_current"] = (100 * t2["excluidos_current"] / t2["excluidos_NA"]).round(1)
t2["pct_con_etiqueta"] = (100 * t2["con_etiqueta"] / t2["total"]).round(1)
t2["tasa_incumplimiento_pct"] = (100 * t2["malos"] / t2["con_etiqueta"]).round(1)
t2.to_csv(RES / "tabla02_registros_por_anio.csv")

# ---- Tabla 4 del informe: partición temporal ----
def conjunto(a):
    return "entrenamiento (hasta 2013)" if a <= 2013 else ("validación (2014)" if a == 2014 else "prueba (2015)")
e = etiq.assign(conjunto=etiq["anio"].map(conjunto))
t3 = e.groupby("conjunto")["objetivo"].agg(registros="count", malos="sum")
t3["malos"] = t3["malos"].astype(int)
t3["pct_del_total"] = (100 * t3["registros"] / t3["registros"].sum()).round(1)
t3["tasa_incumplimiento_pct"] = (100 * t3["malos"] / t3["registros"]).round(1)
t3 = t3.loc[["entrenamiento (hasta 2013)", "validación (2014)", "prueba (2015)"]]
t3.to_csv(RES / "tabla04_particion_temporal.csv")

# ---- Tabla 3 del informe: variables por categoría ----
prop = pd.read_csv(RES / "propuesta_variables_pd.csv")
t4 = prop["categoria"].value_counts().rename("columnas").to_frame()
t4.index.name = "categoria"
t4.to_csv(RES / "tabla03_variables_por_categoria.csv")

def base(ax):
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(colors=TINTA_SUAVE, length=0)
    ax.xaxis.label.set_color(TINTA_SUAVE)

# ---- Figura 1: registros por categoría de loan_status ----
orden = t1.sort_values("registros")
col = {"0": AZUL, "1": NARANJA, "NA (se excluye)": GRIS}
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(orden.index, orden["registros"], color=[col[c] for c in orden["codificacion"]], height=0.62, edgecolor="white")
for y, v in enumerate(orden["registros"]):
    ax.text(v + 6000, y, f"{v:,}".replace(",", "."), va="center", fontsize=8.5, color=TINTA)
ax.set_xlim(0, orden["registros"].max() * 1.15)
ax.xaxis.set_major_formatter(lambda x, _: f"{int(x):,}".replace(",", "."))
ax.set_xlabel("Número de créditos")
ax.set_title("Créditos por estado (loan_status) y su codificación", loc="left", fontsize=11, color=TINTA)
base(ax)
ax.tick_params(axis="y", labelsize=8.5, colors=TINTA)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=AZUL, label="Bueno (0)"), Patch(color=NARANJA, label="Malo (1)"),
                   Patch(color=GRIS, label="NA (excluido)")], frameon=False, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig01_loan_status.png", dpi=200)
plt.close(fig)

# ---- Figura 2: créditos por año de originación según su uso ----
anios = t2.index.astype(str)
buenos = (t2["con_etiqueta"] - t2["malos"]).values
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(anios, buenos, color=AZUL, label="Bueno (0)", width=0.62, edgecolor="white")
ax.bar(anios, t2["malos"].values, bottom=buenos, color=NARANJA, label="Malo (1)", width=0.62, edgecolor="white")
ax.bar(anios, t2["excluidos_NA"].values, bottom=t2["con_etiqueta"].values, color=GRIS, label="NA (excluido)",
       width=0.62, edgecolor="white")
ax.yaxis.set_major_formatter(lambda y, _: f"{int(y/1000)} mil" if y else "0")
ax.set_ylabel("Número de créditos", color=TINTA_SUAVE)
ax.set_title("Créditos por año de originación: etiquetados y excluidos", loc="left", fontsize=11, color=TINTA)
ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
ax.set_axisbelow(True)
base(ax)
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig02_etiquetas_por_anio.png", dpi=200)
plt.close(fig)

for nombre, t in [("Tabla 1", t1), ("Tabla 2", t2), ("Tabla 4 (partición)", t3), ("Tabla 3 (variables)", t4)]:
    print(f"\n{nombre}\n{t.to_string()}")
