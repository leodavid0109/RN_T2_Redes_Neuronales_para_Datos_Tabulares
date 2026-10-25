"""Reporta el balance de clases en las particiones PD ya guardadas. No escribe archivos."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "processed" / "pd.parquet"

df = pd.read_parquet(DATOS, columns=["objetivo", "split_random", "split_temporal"])
if df["objetivo"].isna().any() or not set(df["objetivo"].unique()).issubset({0, 1}):
    raise ValueError("El parquet debe contener solo etiquetas PD 0/1, sin valores faltantes.")

for nombre, columna in (("aleatoria", "split_random"), ("temporal", "split_temporal")):
    print(f"\nPartición {nombre} ({columna})")
    resumen = (
        df.groupby(columna, sort=False)["objetivo"]
        .agg(registros="size", clase_1="sum", prevalencia_clase_1="mean")
        .rename_axis("conjunto")
    )
    resumen["clase_0"] = resumen["registros"] - resumen["clase_1"]
    resumen["porcentaje_clase_0"] = 100 * resumen["clase_0"] / resumen["registros"]
    resumen["porcentaje_clase_1"] = 100 * resumen["prevalencia_clase_1"]
    print(resumen[["registros", "clase_0", "porcentaje_clase_0", "clase_1", "porcentaje_clase_1"]].to_string(float_format=lambda x: f"{x:.2f}"))
