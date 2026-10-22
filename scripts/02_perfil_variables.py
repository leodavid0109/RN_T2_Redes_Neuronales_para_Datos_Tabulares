"""Perfil de cada columna del CSV, calculado sobre los registros con etiqueta (0/1).

Solo lectura. Guarda results/perfil_variables.csv.
Uso:  python scripts/02_perfil_variables.py
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.objetivo import codificar_pd  # noqa: E402

df = pd.read_csv(RAIZ / "data" / "raw" / "loan.csv", low_memory=False)
df["fecha_orig"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
et = df[codificar_pd(df["loan_status"]).notna()]

filas = []
for c in df.columns.drop("fecha_orig"):
    s = et[c]
    nn = s.dropna()
    primera = et.loc[s.notna(), "fecha_orig"].min()
    filas.append({
        "variable": c,
        "tipo": str(s.dtype),
        "pct_faltantes": round(100 * s.isna().mean(), 2),
        "n_unicos": int(nn.nunique()),
        "valor_mas_frecuente": (str(nn.mode().iloc[0])[:40] if len(nn) else ""),
        "pct_valor_mas_frecuente": round(100 * (nn == nn.mode().iloc[0]).mean(), 2) if len(nn) else 0.0,
        "primera_fecha_con_dato": str(primera.date()) if pd.notna(primera) else "",
    })
perfil = pd.DataFrame(filas)
perfil.to_csv(RAIZ / "results" / "perfil_variables.csv", index=False)
pd.set_option("display.width", 200)
print(f"Registros con etiqueta: {len(et)}")
print(perfil.to_string(index=False))
