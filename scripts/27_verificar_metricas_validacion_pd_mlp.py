"""Comprueba que cada checkpoint PD reproduce sus métricas guardadas de validación.

Solo carga filas de validación. Es una verificación de integridad, no un entrenamiento.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def metrics(y, p):
    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr)),
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/pd_mlp_interpretabilidad_v2")
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    target = out / "verificacion_metricas_por_semilla.csv"
    summary_path = out / "verificacion_metricas_resumen.json"
    if target.exists() or summary_path.exists():
        raise FileExistsError("No sobrescribo la verificación; use otra carpeta de salida.")
    pdf = pd.read_parquet(ROOT / "data/processed/pd.parquet")
    pre = joblib.load(ROOT / "models/preprocessor.joblib")
    saved = pd.read_csv(ROOT / "results/pd_mlp_final_particionado/pd_mlp_metricas_por_semilla.csv")
    cols = list(pre.feature_names_in_)
    rows = []
    for part in ("aleatoria", "temporal"):
        split = "split_random" if part == "aleatoria" else "split_temporal"
        valid = pdf.loc[pdf[split].eq("validacion")].reset_index(drop=True)
        y = valid.objetivo.to_numpy(dtype=np.int8)
        xt = pre.transform(valid[cols])
        X = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
        X = np.ascontiguousarray(X, dtype=np.float32)
        for seed in range(30):
            path = ROOT / "models/pd_mlp_final_particionado" / f"pd_mlp_{part}_final_seed_{seed}.pt"
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if ck["seed"] != seed or ck["partition"] != part or ck["probability_of"] != 1:
                raise ValueError(f"Metadatos inesperados en {path}")
            if list(ck["input_columns"]) != cols or ck["input_dim"] != X.shape[1]:
                raise ValueError(f"Contrato de entradas inesperado en {path}")
            from src.mlp_pd import MLPBinario, probabilidades_clase_1
            model = MLPBinario(ck["input_dim"], ck["model_config"])
            model.load_state_dict(ck["state_dict"])
            p = probabilidades_clase_1(model, X)
            actual = metrics(y, p)
            expected_row = saved.loc[(saved.semilla == seed) & (saved.particion == part) &
                                     (saved.conjunto == "validacion")]
            if len(expected_row) != 1:
                raise ValueError(f"No hay una métrica guardada única para {part}/{seed}")
            row = {"particion": part, "semilla": seed, "n_validacion": len(y)}
            for name, value in actual.items():
                expected = float(expected_row.iloc[0][name])
                row[f"{name}_recalculada"] = value
                row[f"{name}_guardada"] = expected
                row[f"{name}_error_abs"] = abs(value - expected)
            rows.append(row)
    result = pd.DataFrame(rows)
    err_cols = [c for c in result if c.endswith("_error_abs")]
    max_error = float(result[err_cols].to_numpy().max())
    result.to_csv(target, index=False)
    summary = {"n_corridas": len(result), "filas_validacion": {
        p: int(result.loc[result.particion == p, "n_validacion"].iloc[0])
        for p in ("aleatoria", "temporal")},
        "tolerancia_absoluta": args.atol, "error_absoluto_maximo": max_error,
        "pasa": bool(max_error <= args.atol),
        "solo_validacion": True, "prueba_consultada_para_metricas": False,
        "parquet_completo_cargado": True}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["pasa"]:
        raise SystemExit("Las métricas recalculadas exceden la tolerancia.")


if __name__ == "__main__":
    main()
