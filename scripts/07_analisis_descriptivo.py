"""Análisis descriptivo y evaluación de las hipótesis H1-H8 (sección 3 del informe).

Usa SOLO el entrenamiento de la partición aleatoria (validación y prueba no se tocan).
Criterio fijado ANTES de explorar:
  Confirmada : diferencia >= 2 puntos porcentuales en la dirección esperada y p < 0,01 (chi-cuadrado 2x2)
  Contraria  : diferencia <= -2 puntos con p < 0,01
  No confirmada: en cualquier otro caso
La diferencia es (tasa del grupo esperado más riesgoso) - (tasa del grupo esperado menos riesgoso).
Uso:  python scripts/07_analisis_descriptivo.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.variables import CATEGORICAS  # noqa: E402

RES, FIG = RAIZ / "results", RAIZ / "figures"
UMBRAL_PP, UMBRAL_P = 2.0, 0.01
NARANJA, TINTA, TINTA_SUAVE = "#D55E00", "#222222", "#666666"

df = pd.read_parquet(RAIZ / "data" / "processed" / "pd.parquet")
tr = df[df["split_random"] == "entrenamiento"].copy()
base = 100 * tr["objetivo"].mean()
print(f"Entrenamiento aleatorio: {len(tr)} registros | tasa base: {base:.2f} %")

# ---------- 1) Estadísticos descriptivos (sin el objetivo) ----------
num = [c for c in tr.columns if c not in ["id", "fecha_orig", "objetivo", "split_random", "split_temporal"]
       and c not in CATEGORICAS and not c.startswith("falta_")]
est = tr[num].describe(percentiles=[0.5]).T[["mean", "std", "min", "50%", "max"]]
est["pct_faltante"] = (100 * tr[num].isna().mean()).round(2)
est = est.rename(columns={"mean": "media", "std": "desv_est", "min": "minimo", "50%": "mediana", "max": "maximo"}).round(2)
est.index.name = "variable"
est.to_csv(RES / "tabla06_estadisticos_descriptivos.csv")

# ---------- 2) Hipótesis ----------
def fmt(x):
    """Número en formato español, sin notación científica."""
    if abs(x) >= 100:
        return f"{x:,.0f}".replace(",", ".")
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def agrupar_cuantiles(serie, q=5):
    cortes = pd.qcut(serie, q, duplicates="drop")
    cats = list(cortes.cat.categories)
    etiquetas = {iv: f"Q{i + 1}" for i, iv in enumerate(cats)}
    rangos = {}
    for i, iv in enumerate(cats):
        izq = serie.min() if i == 0 else iv.left   # el primer intervalo de qcut es abierto a la izquierda
        rangos[etiquetas[iv]] = f"{fmt(izq)} a {fmt(iv.right)}"
    return cortes.map(etiquetas).astype("category"), rangos

tr["g_dti"], r_dti = agrupar_cuantiles(tr["dti"])
tr["g_util"], r_util = agrupar_cuantiles(tr["revol_util"])
tr["g_ing"], r_ing = agrupar_cuantiles(tr["annual_inc"])
tr["g_ant"], r_ant = agrupar_cuantiles(tr["antiguedad_cred_meses"])
tr["g_term"] = tr["term_meses"].map({36.0: "36 meses", 60.0: "60 meses"})
tr["g_delinq"] = np.where(tr["delinq_2yrs"].isna(), None, np.where(tr["delinq_2yrs"] >= 1, "1 o más", "0"))
tr["g_inq"] = np.where(tr["inq_last_6mths"].isna(), None, np.where(tr["inq_last_6mths"] >= 3, "3 o más", tr["inq_last_6mths"].fillna(0).astype(int).astype(str)))
tr["g_viv"] = tr["home_ownership"].replace({"ANY": "Otras", "NONE": "Otras", "OTHER": "Otras"})

# (id, variable, columna de grupo, orden de menos a más riesgoso ESPERADO, texto de la hipótesis, rangos)
H = [
    ("H1", "dti", "g_dti", ["Q1", "Q2", "Q3", "Q4", "Q5"], "A mayor razón deuda/ingreso, mayor probabilidad de incumplimiento", r_dti),
    ("H2", "term", "g_term", ["36 meses", "60 meses"], "Los créditos a 60 meses incumplen más que los de 36 meses", {}),
    ("H3", "revol_util", "g_util", ["Q1", "Q2", "Q3", "Q4", "Q5"], "A mayor utilización del crédito rotativo, mayor riesgo", r_util),
    ("H4", "delinq_2yrs", "g_delinq", ["0", "1 o más"], "Quien tuvo moras de 30 o más días en los últimos 2 años incumple más", {}),
    ("H5", "inq_last_6mths", "g_inq", ["0", "1", "2", "3 o más"], "A más consultas de crédito recientes, mayor riesgo", {}),
    ("H6", "annual_inc", "g_ing", ["Q5", "Q4", "Q3", "Q2", "Q1"], "A mayor ingreso anual, menor riesgo", r_ing),
    ("H7", "antiguedad_cred_meses", "g_ant", ["Q5", "Q4", "Q3", "Q2", "Q1"], "A mayor antigüedad crediticia, menor riesgo", r_ant),
    ("H8", "home_ownership", "g_viv", ["MORTGAGE", "RENT"], "Quien alquila su vivienda incumple más que quien tiene hipoteca", {}),
]

detalle, resultados = [], []
for hid, var, col, orden, texto, rangos in H:
    t = tr.dropna(subset=[col]).groupby(col, observed=True)["objetivo"].agg(n="count", malos="sum")
    t["tasa_pct"] = 100 * t["malos"] / t["n"]
    for g, fila in t.iterrows():
        detalle.append({"hipotesis": hid, "variable": var, "grupo": g, "rango": rangos.get(g, ""),
                        "n": int(fila["n"]), "malos": int(fila["malos"]), "tasa_pct": round(fila["tasa_pct"], 2)})
    menos, mas = orden[0], orden[-1]
    a, b = t.loc[menos], t.loc[mas]
    chi = chi2_contingency([[a["malos"], a["n"] - a["malos"]], [b["malos"], b["n"] - b["malos"]]])
    dif = b["tasa_pct"] - a["tasa_pct"]
    p = chi[1]
    if dif >= UMBRAL_PP and p < UMBRAL_P:
        veredicto = "Confirmada"
    elif dif <= -UMBRAL_PP and p < UMBRAL_P:
        veredicto = "Contraria"
    else:
        veredicto = "No confirmada"
    seq = [t.loc[g, "tasa_pct"] for g in orden if g in t.index]
    monotona = "sí" if all(np.diff(seq) >= 0) else "no"
    resultados.append({"hipotesis": hid, "variable": var, "texto": texto, "grupo_menos_riesgoso": menos,
                       "tasa_menos_pct": round(a["tasa_pct"], 2), "grupo_mas_riesgoso": mas,
                       "tasa_mas_pct": round(b["tasa_pct"], 2), "diferencia_pp": round(dif, 2),
                       "p_valor": float(f"{p:.3g}"), "p_texto": ("< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")), "monotona": monotona if len(orden) > 2 else "n/a",
                       "veredicto_descriptivo": veredicto})

pd.DataFrame(detalle).to_csv(RES / "tasas_por_grupo_hipotesis.csv", index=False)
res = pd.DataFrame(resultados)
res.to_csv(RES / "tabla07_resultados_hipotesis.csv", index=False)
pd.DataFrame([{"hipotesis": h[0], "variable": h[1], "hipotesis_texto": h[4]} for h in H]).assign(
    criterio=f"Confirmada: diferencia >= {UMBRAL_PP:g} pp en la dirección esperada y p < {UMBRAL_P:g}").to_csv(
    RES / "tabla05_hipotesis.csv", index=False)

# ---------- 3) Correlaciones entre variables numéricas ----------
corr = tr[num].corr()
pares = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(num) for b in num[i + 1:] if abs(corr.loc[a, b]) >= 0.7]
pd.DataFrame(pares, columns=["variable_1", "variable_2", "correlacion"]).to_csv(RES / "correlaciones_altas.csv", index=False)

# ---------- 4) Figuras ----------
det = pd.DataFrame(detalle)
def panel(ax, hid, titulo, orden_graf):
    d = det[det.hipotesis == hid].set_index("grupo").loc[orden_graf]
    ax.bar(range(len(d)), d["tasa_pct"], color=NARANJA, width=0.6, edgecolor="white")
    ax.axhline(base, color="#888888", linestyle="--", linewidth=1)
    for i, (v, n) in enumerate(zip(d["tasa_pct"], d["n"])):
        ax.text(i, v + 0.5, f"{v:.1f} %".replace(".", ","), ha="center", fontsize=8.5, color=TINTA,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85))
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(orden_graf, fontsize=8.5, color=TINTA)
    ax.set_title(titulo, loc="left", fontsize=9.5, color=TINTA)
    ax.set_ylim(0, max(det["tasa_pct"].max() * 1.18, 30))
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="y", colors=TINTA_SUAVE, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)

def figura(paneles, archivo, xlabel_comun):
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True)
    for ax, (hid, titulo, orden) in zip(axs.ravel(), paneles):
        panel(ax, hid, titulo, orden)
    for ax in axs[:, 0]:
        ax.set_ylabel("Tasa de incumplimiento (%)", color=TINTA_SUAVE, fontsize=9)
    fig.text(0.5, 0.005, xlabel_comun + "  La línea punteada marca la tasa base (" + f"{base:.1f}".replace(".", ",") + " %).", ha="center", fontsize=8.5, color=TINTA_SUAVE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / archivo, dpi=200)
    plt.close(fig)

figura([("H1", "H1. dti (quintiles)", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        ("H3", "H3. revol_util (quintiles)", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        ("H6", "H6. annual_inc (quintiles)", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        ("H7", "H7. antigüedad crediticia (quintiles)", ["Q1", "Q2", "Q3", "Q4", "Q5"])],
       "fig03_hipotesis_por_quintil.png", "Q1 = quintil más bajo de la variable, Q5 = más alto.")
figura([("H2", "H2. plazo (term)", ["36 meses", "60 meses"]),
        ("H4", "H4. moras en 2 años (delinq_2yrs)", ["0", "1 o más"]),
        ("H5", "H5. consultas en 6 meses (inq_last_6mths)", ["0", "1", "2", "3 o más"]),
        ("H8", "H8. vivienda (home_ownership)", list(det[det.hipotesis == "H8"].sort_values("n", ascending=False)["grupo"]))],
       "fig04_hipotesis_por_grupo.png", "Grupos de cada variable.")

pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_colwidth", 70)
print("\n=== Estadísticos descriptivos ===\n", est.to_string())
print("\n=== Detalle por grupo ===\n", det.to_string(index=False))
print("\n=== Resultados ===\n", res.drop(columns=["texto"]).to_string(index=False))
print("\n=== Pares con |correlación| >= 0,7 ===\n", pd.DataFrame(pares, columns=["v1", "v2", "r"]).to_string(index=False) if pares else "ninguno")
