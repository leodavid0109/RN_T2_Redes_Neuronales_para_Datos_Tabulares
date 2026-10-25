"""Pérdida esperada por crédito: E[L] = PD · LGD · EAD.

  PD   : regresión logística de la Parte 1 (models/pd_logreg_<particion>.joblib).
  LGD  : modelo de la Parte 2 (models/lgd_lineal_<particion>.joblib), recortado a [0, 1].
         Se aplica a TODOS los créditos como pérdida condicionada al incumplimiento,
         aunque se entrenó solo con créditos castigados.
  EAD  : monto financiado al originar (funded_amnt). Es lo que el banco conoce al decidir;
         es conservador porque el crédito se amortiza antes de un posible incumplimiento.

El script contrasta E[L] con la pérdida realizada por deciles de PD (en el conjunto de prueba).
La pérdida realizada de un incumplido es LGD_obs · EAD_al_incumplir (src/lgd.py). Los incumplidos no
castigados (Default, Late 31-120) tienen recuperaciones aún no observadas (censura), por eso se
reportan dos cotas:
  - perdida_realizada_max: los censurados cuentan como pérdida total de su EAD (LGD = 1).
  - perdida_realizada_min: los censurados cuentan como pérdida 0.

Uso (desde la raíz del repo):
    python src/expected_loss.py
    python src/expected_loss.py --particion temporal
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]


def cargar_ead(ruta_ead, ruta_raw):
    """funded_amnt por crédito. Si no existe el archivo procesado, se genera desde loan.csv."""
    if not ruta_ead.exists():
        print(f"Generando {ruta_ead.name} desde {ruta_raw.name} (solo la primera vez)...")
        raw = pd.read_csv(ruta_raw, usecols=["id", "funded_amnt"], low_memory=False)
        raw["id"] = pd.to_numeric(raw["id"], errors="coerce")
        raw = raw.dropna(subset=["id"]).astype({"id": "int64"})
        raw.to_parquet(ruta_ead, index=False)
    return pd.read_parquet(ruta_ead)


def main(a):
    t0 = time.perf_counter()
    part = a.particion
    col_split = "split_random" if part == "aleatoria" else "split_temporal"

    pdf = pd.read_parquet(RAIZ / "data/processed/pd.parquet")
    lgdf = pd.read_parquet(RAIZ / "data/processed/lgd.parquet",
                           columns=["id", "lgd", "ead", "castigado"])
    ead = cargar_ead(RAIZ / a.ead_archivo, RAIZ / a.raw)

    df = pdf.merge(ead, on="id", how="left", validate="one_to_one")
    n_sin_ead = int(df["funded_amnt"].isna().sum())
    if n_sin_ead:
        raise SystemExit(f"{n_sin_ead} créditos sin funded_amnt: revisar el cruce por id.")

    # Predicciones: mismo preprocesador compartido, solo transform
    pre = joblib.load(RAIZ / "models/preprocessor.joblib")
    X = pre.transform(df[list(pre.feature_names_in_)])
    m_pd = joblib.load(RAIZ / f"models/pd_logreg_{part}.joblib")
    m_lgd = joblib.load(RAIZ / f"models/lgd_lineal_{part}.joblib")
    df["pd_hat"] = m_pd.predict_proba(X)[:, 1]
    df["lgd_hat"] = np.clip(m_lgd.predict(X), 0.0, 1.0)
    df["ead_usado"] = df["funded_amnt"]
    df["perdida_esperada"] = df["pd_hat"] * df["lgd_hat"] * df["ead_usado"]

    # Pérdida realizada (solo existe para incumplidos con EAD > 0)
    df = df.merge(lgdf.rename(columns={"lgd": "lgd_obs", "ead": "ead_incumplimiento"}),
                  on="id", how="left", validate="one_to_one")
    perdida = (df["lgd_obs"] * df["ead_incumplimiento"]).fillna(0.0)
    censurado = (df["objetivo"] == 1) & (df["castigado"] != True)  # noqa: E712
    df["perdida_realizada_max"] = np.where(df["objetivo"] == 1, perdida, 0.0)
    df["perdida_realizada_min"] = np.where(censurado, 0.0, df["perdida_realizada_max"])

    # Guardar por crédito (para la app y las figuras)
    sal = df[["id", "fecha_orig", col_split, "objetivo", "pd_hat", "lgd_hat", "ead_usado",
              "perdida_esperada", "perdida_realizada_max", "perdida_realizada_min"]].copy()
    sal = sal.rename(columns={col_split: "conjunto"})
    sal["particion"] = part
    for c in ["pd_hat", "lgd_hat", "ead_usado", "perdida_esperada",
              "perdida_realizada_max", "perdida_realizada_min"]:
        sal[c] = sal[c].astype("float32")
    (RAIZ / "results").mkdir(exist_ok=True)
    sal.to_parquet(RAIZ / f"results/perdida_esperada_{part}.parquet", index=False)

    # Contraste por deciles de PD en el conjunto elegido (por defecto, prueba)
    ev = df[df[col_split] == a.conjunto].copy()
    ev["decil_pd"] = pd.qcut(ev["pd_hat"].rank(method="first"), 10, labels=False) + 1
    g = ev.groupby("decil_pd").agg(
        n=("id", "size"),
        pd_media=("pd_hat", "mean"),
        tasa_incumplimiento_obs=("objetivo", "mean"),
        lgd_hat_media=("lgd_hat", "mean"),
        ead_media=("ead_usado", "mean"),
        perdida_esperada_total=("perdida_esperada", "sum"),
        perdida_realizada_min_total=("perdida_realizada_min", "sum"),
        perdida_realizada_max_total=("perdida_realizada_max", "sum"),
    ).reset_index()
    g["razon_esperada_vs_max"] = g["perdida_esperada_total"] / g["perdida_realizada_max_total"]
    g["razon_esperada_vs_min"] = g["perdida_esperada_total"] / g["perdida_realizada_min_total"]
    g.to_csv(RAIZ / f"results/perdida_esperada_deciles_{part}.csv", index=False)

    tot = {
        "particion": part, "conjunto": a.conjunto, "n": int(len(ev)),
        "ead_definicion": "funded_amnt al originar",
        "perdida_esperada_total": float(ev["perdida_esperada"].sum()),
        "perdida_realizada_min_total": float(ev["perdida_realizada_min"].sum()),
        "perdida_realizada_max_total": float(ev["perdida_realizada_max"].sum()),
        "exposicion_total_funded": float(ev["ead_usado"].sum()),
        "pd_media": float(ev["pd_hat"].mean()),
        "tasa_incumplimiento_obs": float(ev["objetivo"].mean()),
        "incumplidos_censurados_en_evaluacion": int(censurado[ev.index].sum()),
        "incumplidos_en_evaluacion": int((ev["objetivo"] == 1).sum()),
        "ead_incumplimiento_sobre_funded_media": float(
            (ev.loc[ev["ead_incumplimiento"] > 0, "ead_incumplimiento"]
             / ev.loc[ev["ead_incumplimiento"] > 0, "funded_amnt"]).mean()),
        "tiempo_s": round(time.perf_counter() - t0, 2),
    }
    json.dump(tot, open(RAIZ / f"results/perdida_esperada_resumen_{part}.json", "w"), indent=2)

    pd.set_option("display.width", 200)
    print(f"Partición {part} | conjunto {a.conjunto} | n = {len(ev):,}")
    print(g.round(4).to_string(index=False))
    print("\nTotales:")
    for k, v in tot.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", choices=["aleatoria", "temporal"], default="aleatoria")
    ap.add_argument("--conjunto", choices=["entrenamiento", "validacion", "prueba"], default="prueba")
    ap.add_argument("--ead-archivo", default="data/processed/ead.parquet")
    ap.add_argument("--raw", default="data/raw/loan.csv")
    main(ap.parse_args())