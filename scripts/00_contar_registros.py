"""Cuenta los registros del CSV original y por categoría de loan_status.

Solo lectura: no modifica los datos. Guarda los conteos en results/.
Uso:  python scripts/00_contar_registros.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.objetivo import MAPA_PD, codificar_pd  # noqa: E402

CSV = RAIZ / "data" / "raw" / "loan.csv"
SALIDA = RAIZ / "results"
SALIDA.mkdir(exist_ok=True)

df = pd.read_csv(CSV, low_memory=False)
df["objetivo"] = codificar_pd(df["loan_status"])
df["fecha_orig"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

# Categorías de loan_status que el enunciado no contempla
no_previstas = sorted(set(df["loan_status"].dropna()) - set(MAPA_PD))
if no_previstas:
    print("ATENCIÓN: categorías no contempladas en la tabla del enunciado:", no_previstas)

# Conteo por categoría original y su codificación
por_estado = (
    df["loan_status"].value_counts(dropna=False).rename("registros").to_frame()
    .assign(codificacion=lambda t: t.index.map(lambda s: MAPA_PD.get(s, "no prevista")))
)
por_estado.index.name = "loan_status"
por_estado.to_csv(SALIDA / "conteo_loan_status.csv")

etiquetados = df[df["objetivo"].notna()]
malos = int((etiquetados["objetivo"] == 1).sum())
buenos = int((etiquetados["objetivo"] == 0).sum())

resumen = {
    "archivo": CSV.name,
    "filas_totales": int(len(df)),
    "columnas_totales": int(df.shape[1] - 2),  # sin las 2 columnas auxiliares agregadas aquí
    "fechas_issue_d_sin_parsear": int(df["fecha_orig"].isna().sum()),
    "periodo_originacion_total": [str(df["fecha_orig"].min().date()), str(df["fecha_orig"].max().date())],
    "registros_con_etiqueta": int(len(etiquetados)),
    "registros_NA_excluidos": int(df["objetivo"].isna().sum()),
    "buenos_0": buenos,
    "malos_1": malos,
    "tasa_incumplimiento_etiquetados": round(malos / len(etiquetados), 6),
    "periodo_originacion_etiquetados": [
        str(etiquetados["fecha_orig"].min().date()), str(etiquetados["fecha_orig"].max().date())],
}
(SALIDA / "conteo_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

# Etiquetados y tasa de incumplimiento por año de originación (útil para elegir la partición temporal)
por_anio = (
    df.assign(anio=df["fecha_orig"].dt.year)
    .groupby("anio")
    .agg(total=("loan_status", "size"),
         etiquetados=("objetivo", "count"),
         malos=("objetivo", "sum"))
    .assign(tasa_malos=lambda t: (t["malos"] / t["etiquetados"]).round(4))
)
por_anio.to_csv(SALIDA / "conteo_por_anio.csv")

print(por_estado.to_string(), "\n")
print(json.dumps(resumen, indent=2, ensure_ascii=False), "\n")
print(por_anio.to_string())
