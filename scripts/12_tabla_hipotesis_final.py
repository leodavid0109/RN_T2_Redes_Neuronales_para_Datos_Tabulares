"""Genera las tablas de hipótesis y añade importancias MLP por partición.

La importancia nueva se lee por (variable, particion) desde el resumen de
permutación agrupada. No promedia ni mezcla los protocolos aleatorio y temporal.
Las salidas se escriben en un directorio nuevo para preservar tablas anteriores.

Ejemplo:
  python scripts/12_tabla_hipotesis_final.py --importance-file results/pd_mlp_interpretabilidad_v2/importancia_agrupada_validacion.csv --output-dir results/pd_mlp_interpretabilidad_v2/tabla_hipotesis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_NEAR_ZERO = 0.02

# Hipótesis -> (nombre de tabla, nombre original, signo esperado, columnas logísticas).
HYPOTHESES = {
    "H1": ("dti", "dti", +1, ["dti"]),
    "H2": ("term", "term", +1, ["term_meses"]),
    "H3": ("revol_util", "revol_util", +1, ["revol_util"]),
    "H4": ("delinq_2yrs", "delinq_2yrs", +1, ["delinq_2yrs"]),
    "H5": ("inq_last_6mths", "inq_last_6mths", +1, ["inq_last_6mths"]),
    "H6": ("annual_inc", "annual_inc", -1, ["annual_inc"]),
    "H7": ("antiguedad_cred_meses", "earliest_cr_line", -1, ["antiguedad_cred_meses"]),
    "H8": ("home_ownership", "home_ownership", +1,
          ["home_ownership_RENT", "home_ownership_MORTGAGE"]),
}

# Correspondencia entre nombre de la hipótesis y los grupos del resumen del MLP.
MLP_GROUP = {"dti": "dti", "term": "term_meses", "revol_util": "revol_util",
             "delinq_2yrs": "delinq_2yrs", "inq_last_6mths": "inq_last_6mths",
             "annual_inc": "annual_inc", "earliest_cr_line": "antiguedad_cred_meses",
             "home_ownership": "home_ownership"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance-file",
                        default="results/pd_mlp_interpretabilidad_v2/importancia_agrupada_validacion.csv")
    parser.add_argument("--output-dir",
                        default="results/pd_mlp_interpretabilidad_v2/tabla_hipotesis")
    args = parser.parse_args()
    imp_path = Path(args.importance_file)
    out_dir = Path(args.output_dir)
    if not imp_path.is_absolute():
        imp_path = ROOT / imp_path
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    outputs = [out_dir / "tabla_hipotesis_final.csv", out_dir / "tabla10_coeficientes_hipotesis.csv"]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"No sobrescribo tablas existentes: {[str(p) for p in outputs]}")

    coef = pd.read_csv(ROOT / "results/regresion_logistica_coeficientes.csv")
    weights = coef.pivot(index="variable", columns="particion", values="coeficiente")
    desc = pd.read_csv(ROOT / "results/tabla07_resultados_hipotesis.csv").set_index("hipotesis")
    importance = pd.read_csv(imp_path) if imp_path.exists() else None
    if importance is not None and {"particion", "variable", "delta_logloss"}.issubset(importance.columns):
        rank = importance.copy()
        rank["rango"] = rank.groupby("particion")["delta_logloss"].rank(ascending=False, method="min")
        imp_pivot = rank.pivot(index="variable", columns="particion", values="delta_logloss")
        rank_pivot = rank.pivot(index="variable", columns="particion", values="rango")
    else:
        imp_pivot = rank_pivot = None

    rows = []
    for hypothesis, (name, original, sign, logistic_cols) in HYPOTHESES.items():
        if hypothesis == "H8":
            random_coef = weights.loc["home_ownership_RENT", "aleatoria"] - weights.loc["home_ownership_MORTGAGE", "aleatoria"]
            temporal_coef = weights.loc["home_ownership_RENT", "temporal"] - weights.loc["home_ownership_MORTGAGE", "temporal"]
            display_name = "home_ownership (RENT menos MORTGAGE)"
        else:
            random_coef = weights.loc[logistic_cols[0], "aleatoria"]
            temporal_coef = weights.loc[logistic_cols[0], "temporal"]
            display_name = name
        same_sign = np.sign(random_coef) == sign and np.sign(temporal_coef) == sign
        near_zero = max(abs(random_coef), abs(temporal_coef)) < THRESHOLD_NEAR_ZERO
        row = {
            "hipotesis": hypothesis, "variable": display_name,
            "signo_esperado": "+" if sign > 0 else "-",
            "coef_aleatoria": random_coef, "coef_temporal": temporal_coef,
            "coincide_logistica": "Sí, casi nulo" if same_sign and near_zero else ("Sí" if same_sign else "No"),
            "diferencia_descriptiva_pp": desc.loc[hypothesis, "diferencia_pp"],
            "resultado_descriptivo": desc.loc[hypothesis, "veredicto_descriptivo"],
        }
        if imp_pivot is not None:
            group = MLP_GROUP[original]
            for part, label in (("aleatoria", "aleatoria"), ("temporal", "temporal")):
                row[f"importancia_mlp_{label}_delta_logloss"] = imp_pivot.loc[group, part]
                row[f"rango_mlp_{label}"] = int(rank_pivot.loc[group, part])
        rows.append(row)

    final = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(outputs[0], index=False)
    final[["hipotesis", "variable", "signo_esperado", "coef_aleatoria", "coef_temporal", "coincide_logistica"]].round(3).to_csv(
        outputs[1], index=False)
    print(final.round(4).to_string(index=False))
    print(f"Tablas guardadas en {out_dir}")
    if imp_pivot is None:
        print("El archivo de importancia no existe o no incluye particion/variable/delta_logloss.")


if __name__ == "__main__":
    main()
