"""Construye la scorecard de la PD sobre la regresión logística (parámetros en configs/scorecard.yaml).

Salidas:
  models/scorecard_referencia.json            medias por variable y puntaje promedio de la población de referencia
  results/score_distribution.csv              puntajes de la población de referencia (para los percentiles de la app)
  results/scorecard_prueba_numerica.csv       prueba: al duplicar las odds el puntaje sube exactamente PDO puntos
  results/scorecard_contribuciones_por_variable.csv   cuánto mueve cada variable el puntaje
  results/scorecard_por_decil.csv             puntaje y tasa de incumplimiento por decil (validación y prueba)
  results/scorecard_resumen.json
  figures/fig07_distribucion_puntaje.png, figures/fig08_contribucion_por_variable.png
Uso:  python scripts/10_scorecard.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.scorecard import (Scorecard, factor_offset, probabilidad_desde_puntaje,  # noqa: E402
                           puntaje_desde_odds, puntaje_desde_probabilidad)

df = pd.read_parquet(RAIZ / "data" / "processed" / "pd.parquet")
cfg = Scorecard._config(RAIZ)
col, conj = cfg["columna_particion"], cfg["conjunto_referencia"]
ref = df[df[col] == conj]
sc = Scorecard.construir(ref, RAIZ)
sc.guardar(RAIZ)
PDO, BASE, ODDS = sc.pdo, sc.puntaje_base, sc.odds_base

# ---- Puntajes de todos los créditos y distribución de referencia ----
df["puntaje"] = sc.puntaje(df)
df["pd"] = sc.probabilidad(df)
ref_p = df.loc[ref.index, ["id", "puntaje"]].copy()
ref_p["puntaje"] = ref_p["puntaje"].round(4)
ref_p.to_csv(RAIZ / "results" / "score_distribution.csv", index=False)
sc.referencia = np.sort(ref_p["puntaje"].to_numpy())

# ---- Comprobaciones ----
# 1) prueba numérica: duplicar las odds suma exactamente PDO puntos (sobre odds reales del modelo)
odds_reales = ((1 - df["pd"]) / df["pd"]).to_numpy()[:1000]
dif = puntaje_desde_odds(2 * odds_reales, PDO, BASE, ODDS) - puntaje_desde_odds(odds_reales, PDO, BASE, ODDS)
assert np.allclose(dif, PDO, rtol=0, atol=1e-9)
odds_demo = np.array([1.0, 3.57, 12.5, 25.0, 50.0, 100.0, 200.0])
prueba = pd.DataFrame({"odds": odds_demo, "pd": 1 / (1 + odds_demo),
                       "puntaje": puntaje_desde_odds(odds_demo, PDO, BASE, ODDS),
                       "puntaje_con_odds_x2": puntaje_desde_odds(2 * odds_demo, PDO, BASE, ODDS)})
prueba["diferencia"] = prueba["puntaje_con_odds_x2"] - prueba["puntaje"]
assert np.allclose(prueba["diferencia"], PDO, atol=1e-9)
prueba.to_csv(RAIZ / "results" / "scorecard_prueba_numerica.csv", index=False)
# 2) el puntaje coincide con la transformación de la probabilidad del modelo
assert np.allclose(df["puntaje"], puntaje_desde_probabilidad(df["pd"], PDO, BASE, ODDS), atol=1e-6)
# 3) puntos aditivos: promedio + contribuciones por variable = puntaje
muestra = df.sample(5000, random_state=0)
contrib_m = sc.contribuciones(muestra[list(sc.pre.feature_names_in_)])
assert np.allclose(sc.puntaje_promedio + contrib_m.sum(axis=1), muestra["puntaje"], atol=1e-8)
assert abs(sc.puntaje_promedio - df.loc[ref.index, "puntaje"].mean()) < 1e-8
# 4) el puntaje conserva el orden de la PD: mismo AUC que el modelo (prueba aleatoria)
prueba_a = df[df[col] == "prueba"]
auc_score = roc_auc_score(prueba_a["objetivo"], -prueba_a["puntaje"])
auc_pd = roc_auc_score(prueba_a["objetivo"], prueba_a["pd"])
metr = pd.read_csv(RAIZ / "results" / "regresion_logistica_metricas.csv")
auc_ref = float(metr[(metr["particion"] == "aleatoria") & (metr["conjunto"] == "prueba")]["auc"].iloc[0])
assert abs(auc_score - auc_pd) < 1e-12 and abs(auc_pd - auc_ref) < 1e-9, (auc_score, auc_pd, auc_ref)

# ---- Contribuciones por variable (población de referencia) ----
c_ref = sc.contribuciones(ref[list(sc.pre.feature_names_in_)])
tab = pd.DataFrame({"media_puntos_sin_centrar": pd.Series(sc.medias), "desv_est": c_ref.std(), "p5": c_ref.quantile(0.05),
                    "p50": c_ref.quantile(0.50), "p95": c_ref.quantile(0.95)})
tab["rango_p5_p95"] = tab["p95"] - tab["p5"]
tab = tab.sort_values("rango_p5_p95", ascending=False)
tab.index.name = "variable"
tab.round(4).to_csv(RAIZ / "results" / "scorecard_contribuciones_por_variable.csv")

# ---- Puntaje y tasa de incumplimiento por decil (decil 1 = menor puntaje = mayor riesgo) ----
filas = []
for nombre in ["validacion", "prueba"]:
    d = df[df[col] == nombre].copy()
    d["decil"] = pd.qcut(d["puntaje"].rank(method="first"), 10, labels=False) + 1
    g = d.groupby("decil").agg(registros=("objetivo", "size"), puntaje_min=("puntaje", "min"), puntaje_medio=("puntaje", "mean"),
                               puntaje_max=("puntaje", "max"), tasa_malos_pct=("objetivo", lambda s: 100 * s.mean()),
                               pd_media_pct=("pd", lambda s: 100 * s.mean())).reset_index()
    g.insert(0, "conjunto", nombre)
    filas.append(g)
dec = pd.concat(filas)
dec.round(3).to_csv(RAIZ / "results" / "scorecard_por_decil.csv", index=False)
monotona = {n: bool(dec[dec["conjunto"] == n]["tasa_malos_pct"].is_monotonic_decreasing) for n in ["validacion", "prueba"]}

# ---- Resumen ----
factor, offset = factor_offset(PDO, BASE, ODDS)
pun = df.loc[ref.index, "puntaje"]
resumen = {"puntaje_base": BASE, "odds_base": ODDS, "pd_en_odds_base": 1 / (1 + ODDS), "pdo": PDO,
           "factor": factor, "offset": offset, "modelo": cfg["modelo"], "poblacion_referencia": f"{col} = {conj}",
           "n_referencia": int(len(ref)), "puntaje_promedio": sc.puntaje_promedio,
           "puntaje_referencia": {"min": float(pun.min()), "p5": float(pun.quantile(.05)), "p25": float(pun.quantile(.25)),
                                  "mediana": float(pun.median()), "p75": float(pun.quantile(.75)), "p95": float(pun.quantile(.95)),
                                  "max": float(pun.max()), "desv_est": float(pun.std())},
           "auc_puntaje_prueba_aleatoria": auc_score, "tasa_malos_monotona_por_decil": monotona,
           "puntos_por_odds_x2_verificado": True,
           "rango_mostrado": [sc.puntaje_minimo, sc.puntaje_maximo],
           "creditos_acotados_por_debajo": int((df["puntaje"] < sc.puntaje_minimo).sum()),
           "creditos_acotados_por_encima": int((df["puntaje"] > sc.puntaje_maximo).sum()),
           "creditos_total": int(len(df))}
(RAIZ / "results" / "scorecard_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

# ---- Figuras ----
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
t = prueba_a
fig, ax = plt.subplots(figsize=(6.4, 4))
lim = (np.floor(t["puntaje"].quantile(0.001)), np.ceil(t["puntaje"].quantile(0.999)))
for v, color, etq in [(0, "#0072B2", "Buenos pagadores"), (1, "#D55E00", "Malos pagadores (incumplen)")]:
    ax.hist(t.loc[t["objetivo"] == v, "puntaje"], bins=60, range=lim, density=True, alpha=0.6, color=color, label=etq)
ax.axvline(sc.puntaje_promedio, color="#8A8A8A", linestyle="--")
ax.text(sc.puntaje_promedio, ax.get_ylim()[1] * 1.0, " promedio", color="#555555", va="top", ha="left", fontsize=9)
ax.set_xlabel("Puntaje de la scorecard")
ax.set_ylabel("Densidad")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(RAIZ / "figures" / "fig07_distribucion_puntaje.png", dpi=200)
plt.close(fig)

top = tab.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 5))
ax.barh(top.index, top["p95"] - top["p5"], left=top["p5"], color="#0072B2")
ax.axvline(0, color="#8A8A8A", linestyle="--")
ax.set_xlabel("Puntos frente al solicitante promedio (p5 a p95)")
fig.tight_layout()
fig.savefig(RAIZ / "figures" / "fig08_contribucion_por_variable.png", dpi=200)
plt.close(fig)

pd.set_option("display.width", 200)
print(json.dumps(resumen, indent=2, ensure_ascii=False))
print(tab.round(2).head(24).to_string())
print(dec[dec["conjunto"] == "prueba"].round(2).to_string(index=False))
print(prueba.round(3).to_string(index=False))
