"""Genera app/datos_app.json: rangos y valores por defecto de las variables que ve el usuario en la app.

La app usa este archivo pequeño en lugar de cargar data/processed/pd.parquet (que no se sube a la nube).
Uso (desde la raiz del repo):
    python scripts/22_preparar_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.variables import CATEGORICAS  # noqa: E402

pre = joblib.load(RAIZ / "models/preprocessor.joblib")
# Las 24 variables de entrada; los indicadores falta_* los calcula Scorecard.preparar
cols = [c for c in pre.feature_names_in_ if not c.startswith("falta_")]
df = pd.read_parquet(RAIZ / "data/processed/pd.parquet", columns=cols)

datos = {}
for c in cols:
    s = df[c]
    if c in CATEGORICAS:
        vc = s.dropna().astype(str).value_counts()
        datos[c] = {"tipo": "cat", "valores": sorted(vc.index.tolist()), "defecto": str(vc.index[0])}
    else:
        x = s.dropna().astype(float)
        datos[c] = {
            "tipo": "num",
            "min": float(x.min()),
            "max": float(x.max()),
            "defecto": float(x.median()),
            "falta": float(s.isna().mean()),
            "valores": sorted(x.unique().tolist()) if x.nunique() <= 6 else None,
            "entero": bool(np.allclose(x, np.round(x))),
        }

(RAIZ / "app").mkdir(exist_ok=True)
with open(RAIZ / "app/datos_app.json", "w", encoding="utf-8") as f:
    json.dump(datos, f, indent=1, ensure_ascii=False)
print(f"app/datos_app.json listo: {len(datos)} variables")