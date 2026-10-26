"""Modelo de referencia de la LGD: regresión lineal con penalización L2 (Ridge).

Usa el preprocesador compartido (solo transform, sin reajustar), la muestra de castigados y
las mismas particiones que PD. La penalización se elige solo con validación; la prueba solo
se reporta. Ridge es determinista: su varianza entre semillas es cero.

Uso:
    python src/lgd_lineal.py
    python src/lgd_lineal.py --alphas 0.1 1 10 --demo
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from metricas_lgd import metricas_lgd

SPLITS = {"aleatoria": "split_random", "temporal": "split_temporal"}
CONJUNTOS = ["entrenamiento", "validacion", "prueba"]


def fila(modelo, particion, conjunto, y, p, params, alpha, tiempo):
    return {"modelo": modelo, "particion": particion, "conjunto": conjunto,
            "alpha": alpha, "n": len(y), **metricas_lgd(y, p),
            "pct_fuera_de_rango": float(np.mean((p < 0) | (p > 1))),
            "params": params, "tiempo_busqueda_s": tiempo}


def main(a):
    t_ini = time.perf_counter()
    df = pd.read_parquet(a.data)
    df = df[df["castigado"]].reset_index(drop=True)
    if a.demo:
        df = df.sample(min(5000, len(df)), random_state=a.seed).reset_index(drop=True)

    pre = joblib.load(a.pre)
    X = pre.transform(df)          # solo transform: el preprocesador NO se reajusta
    y = df["lgd"].to_numpy()
    print(f"Muestra: {len(df):,} castigados | {X.shape[1]} características")

    # El modo demo escribe en models/demo y results/demo: nunca pisa los resultados finales
    sub = "demo" if a.demo else ""
    dir_mod, dir_res = Path("models") / sub, Path("results") / sub
    dir_mod.mkdir(parents=True, exist_ok=True)
    dir_res.mkdir(parents=True, exist_ok=True)
    filas, preds = [], []

    for part, col in SPLITS.items():
        idx = {c: np.where(df[col] == c)[0] for c in CONJUNTOS}
        tr, va = idx["entrenamiento"], idx["validacion"]

        # Referencia trivial: predecir la media de entrenamiento
        media = y[tr].mean()
        for c in CONJUNTOS:
            filas.append(fila("media_train", part, c, y[idx[c]],
                              np.full(len(idx[c]), media), 1, None, 0.0))

        # Búsqueda de alpha solo con validación
        t0 = time.perf_counter()
        mejor = None
        for alpha in a.alphas:
            mod = Ridge(alpha=alpha).fit(X[tr], y[tr])
            rmse_va = metricas_lgd(y[va], mod.predict(X[va]))["rmse"]
            if mejor is None or rmse_va < mejor[0]:
                mejor = (rmse_va, alpha, mod)
        t_busq = time.perf_counter() - t0
        _, alpha, mod = mejor
        params = int(mod.coef_.size + 1)   # coeficientes + intercepto
        print(f"[{part}] alpha elegido = {alpha} | {len(a.alphas)} configuraciones | "
              f"{t_busq:.2f} s | {params} parámetros")

        for c in CONJUNTOS:
            p = mod.predict(X[idx[c]])
            filas.append(fila("ridge", part, c, y[idx[c]], p, params, alpha, t_busq))
            preds.append(pd.DataFrame({
                "id": df.loc[idx[c], "id"].to_numpy(), "particion": part,
                "conjunto": c, "lgd_obs": y[idx[c]],
                "lgd_pred_cruda": p, "lgd_pred": np.clip(p, 0, 1)}))
        joblib.dump(mod, dir_mod / f"lgd_lineal_{part}.joblib")

    res = pd.DataFrame(filas)
    res.to_csv(dir_res / "lgd_lineal.csv", index=False)
    pd.concat(preds).to_parquet(dir_res / "lgd_lineal_pred.parquet", index=False)

    t_total = time.perf_counter() - t_ini
    json.dump({"configuraciones_por_particion": len(a.alphas),
               "tiempo_total_s": round(t_total, 2), "demo": a.demo},
              open(dir_res / "lgd_lineal_resumen.json", "w"), indent=2)

    cols = ["modelo", "particion", "conjunto", "mae", "rmse", "sesgo", "r2",
            "pct_fuera_de_rango"]
    print("\n", res[res["conjunto"] != "entrenamiento"][cols].round(4).to_string(index=False))
    print(f"\nTiempo total: {t_total:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/lgd.parquet")
    ap.add_argument("--pre", default="models/preprocessor.joblib")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.01, 0.1, 1, 10, 100, 1000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    main(ap.parse_args())