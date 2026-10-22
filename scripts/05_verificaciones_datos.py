"""Verifica con los datos las afirmaciones del apartado de datos que no salen de otras tablas.

Solo lectura. Guarda results/verificaciones_datos.csv.
Uso:  python scripts/05_verificaciones_datos.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.objetivo import codificar_pd  # noqa: E402
from src.variables import VARIABLES  # noqa: E402

df = pd.read_csv(RAIZ / "data" / "raw" / "loan.csv", low_memory=False)
df["objetivo"] = codificar_pd(df["loan_status"])
df["fecha"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
et = df[df["objetivo"].notna()]
filas = []


def reg(grupo, afirmacion, valor):
    filas.append({"grupo": grupo, "afirmacion": afirmacion, "valor": valor})


# 1) Variables "casi vacías": cuántos registros etiquetados SÍ tienen dato
casi_vacias = [v for v, (c, _) in VARIABLES.items() if c == "excluir_faltantes"]
for v in casi_vacias:
    reg("faltantes", f"{v}: registros etiquetados con dato (de {len(et)})", int(et[v].notna().sum()))

# 2) Faltantes "estructurales": ¿el vacío ocurre solo antes de agosto de 2012?
corte = pd.Timestamp("2012-08-01")
for v in ["tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "mths_since_last_major_derog"]:
    reg("faltantes", f"{v}: % faltante antes de ago-2012", round(100 * et.loc[et.fecha < corte, v].isna().mean(), 2))
    reg("faltantes", f"{v}: % faltante desde ago-2012", round(100 * et.loc[et.fecha >= corte, v].isna().mean(), 2))

# 3) funded_amnt frente a loan_amnt
reg("redundancia", "% de créditos (todos) con funded_amnt == loan_amnt", round(100 * (df.loan_amnt == df.funded_amnt).mean(), 2))
reg("redundancia", "% de etiquetados con funded_amnt == loan_amnt", round(100 * (et.loan_amnt == et.funded_amnt).mean(), 2))

# 4) Variables posteriores: ejemplos usados en el texto
buenos = et[et.objetivo == 0]
reg("posteriores", "% de buenos con total_rec_prncp == funded_amnt (|dif| <= 0,01)",
    round(100 * ((buenos.total_rec_prncp - buenos.funded_amnt).abs() <= 0.01).mean(), 2))
con_rec = et[et.recoveries > 0]
reg("posteriores", "etiquetados con recoveries > 0", int(len(con_rec)))
reg("posteriores", "  de ellos, % que son malos (objetivo = 1)", round(100 * (con_rec.objetivo == 1).mean(), 2))
reg("posteriores", "buenos (objetivo = 0) con recoveries > 0", int(((et.objetivo == 0) & (et.recoveries > 0)).sum()))

# 5) installment depende de la tasa: cuota de amortización con monto, tasa y plazo
n = df["term"].str.extract(r"(\d+)")[0].astype(float)
r = df["int_rate"] / 1200
pmt = df["loan_amnt"] * r / (1 - (1 + r) ** (-n))
reg("tasa y cuota", "% de créditos con |installment - cuota calculada(loan_amnt, int_rate, term)| <= 0,02",
    round(100 * ((df["installment"] - pmt).abs() <= 0.02).mean(), 2))
for g, sub in df.groupby("grade"):
    reg("tasa y cuota", f"int_rate medio, grado {g} (mín–máx)",
        f"{sub.int_rate.mean():.2f} ({sub.int_rate.min():.2f}–{sub.int_rate.max():.2f})")

# 6) Constantes
reg("constantes", "policy_code: valores distintos en etiquetados", sorted(et["policy_code"].dropna().unique().tolist()))
reg("constantes", "application_type: conteo en etiquetados", et["application_type"].value_counts().to_dict())
reg("constantes", "pymnt_plan: conteo en etiquetados", et["pymnt_plan"].value_counts().to_dict())

out = pd.DataFrame(filas)
out.to_csv(RAIZ / "results" / "verificaciones_datos.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 120)
print(out.to_string(index=False))
