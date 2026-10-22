"""Genera los datos procesados de la Parte 1 (PD): características, objetivo, particiones y preprocesador.

Salidas:
  data/processed/pd.parquet                 características + objetivo + split_random + split_temporal
  models/preprocessor.joblib                preprocesador ajustado (un solo preprocesador)
  results/verificacion_datos_procesados.json   comprobaciones de la salida
Uso:  python scripts/06_generar_datos_pd.py
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.objetivo import codificar_pd  # noqa: E402
from src.variables import CATEGORICAS, INDICADOR_FALTANTE, USADAS, VARIABLES  # noqa: E402

# ---------------- Parámetros (versión v0; cambiarlos exige avisar al equipo) ----------------
SEMILLA = 42
PROPORCION_ALEATORIA = (0.70, 0.15, 0.15)       # entrenamiento, validación, prueba
ULTIMO_ANIO_ENTRENAMIENTO, ANIO_VALIDACION, ANIO_PRUEBA = 2013, 2014, 2015
# Con qué registros se ajusta el único preprocesador:
#   "entrenamiento_aleatorio": decisión del equipo (incluye créditos de 2014-2015 que en la partición
#                              temporal son validación/prueba: fuga leve que se declara en el informe)
#   "interseccion":            entrenamiento aleatorio Y hasta 2013 (sin fuga hacia ninguna partición)
AJUSTE_PREPROCESADOR = "entrenamiento_aleatorio"
# --------------------------------------------------------------------------------------------

df = pd.read_csv(RAIZ / "data" / "raw" / "loan.csv", usecols=["id", "issue_d", "loan_status"] + USADAS, low_memory=False)
n_total = len(df)
df["objetivo"] = codificar_pd(df["loan_status"])
df = df[df["objetivo"].notna()].copy()
df["objetivo"] = df["objetivo"].astype(int)
df["fecha_orig"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

# ---- Transformaciones ----
cred = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
df["antiguedad_cred_meses"] = (df["fecha_orig"].dt.year - cred.dt.year) * 12 + (df["fecha_orig"].dt.month - cred.dt.month)


def emp_a_numero(valor):
    if pd.isna(valor):
        return np.nan
    if str(valor).startswith("<"):          # "< 1 year"
        return 0.0
    digitos = re.findall(r"\d+", str(valor))  # "10+ years" -> 10 ; "n/a" -> sin dígitos
    return float(digitos[0]) if digitos else np.nan


df["emp_length_anios"] = df["emp_length"].map(emp_a_numero)
df["term_meses"] = df["term"].str.extract(r"(\d+)")[0].astype(float)
for v in INDICADOR_FALTANTE:
    df[f"falta_{v}"] = df[v].isna().astype(int)

NUMERICAS = (["loan_amnt", "term_meses", "emp_length_anios", "annual_inc", "dti", "delinq_2yrs",
              "antiguedad_cred_meses", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
              "total_acc", "acc_now_delinq", "collections_12_mths_ex_med", "mths_since_last_delinq",
              "mths_since_last_record", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim"]
             + [f"falta_{v}" for v in INDICADOR_FALTANTE])
CARACTERISTICAS = NUMERICAS + CATEGORICAS

# Las 24 variables decididas deben ser exactamente el origen de las características
origen = {"term": "term_meses", "emp_length": "emp_length_anios", "earliest_cr_line": "antiguedad_cred_meses"}
derivadas_de = {origen.get(v, v) for v in USADAS}
assert derivadas_de == set(NUMERICAS[:20]) | set(CATEGORICAS), derivadas_de ^ (set(NUMERICAS[:20]) | set(CATEGORICAS))
assert len(USADAS) == 24

# ---- Particiones ----
idx = np.arange(len(df))
y = df["objetivo"].to_numpy()
tr, resto = train_test_split(idx, test_size=1 - PROPORCION_ALEATORIA[0], stratify=y, random_state=SEMILLA)
va, te = train_test_split(resto, test_size=PROPORCION_ALEATORIA[2] / (PROPORCION_ALEATORIA[1] + PROPORCION_ALEATORIA[2]),
                          stratify=y[resto], random_state=SEMILLA)
df["split_random"] = ""
df.iloc[tr, df.columns.get_loc("split_random")] = "entrenamiento"
df.iloc[va, df.columns.get_loc("split_random")] = "validacion"
df.iloc[te, df.columns.get_loc("split_random")] = "prueba"

anio = df["fecha_orig"].dt.year
df["split_temporal"] = np.select(
    [anio <= ULTIMO_ANIO_ENTRENAMIENTO, anio == ANIO_VALIDACION, anio == ANIO_PRUEBA],
    ["entrenamiento", "validacion", "prueba"], default="SIN_ASIGNAR")
assert (df["split_random"] != "").all() and (df["split_temporal"] != "SIN_ASIGNAR").all()

# ---- Preprocesador (un solo objeto) ----
if AJUSTE_PREPROCESADOR == "entrenamiento_aleatorio":
    mascara = df["split_random"] == "entrenamiento"
elif AJUSTE_PREPROCESADOR == "interseccion":
    mascara = (df["split_random"] == "entrenamiento") & (df["split_temporal"] == "entrenamiento")
else:
    raise ValueError(AJUSTE_PREPROCESADOR)

pre = ColumnTransformer(
    [("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), NUMERICAS),
     ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS)],
    verbose_feature_names_out=False)
assert df[CATEGORICAS].notna().all().all(), "hay categóricas con valores vacíos"
pre.fit(df.loc[mascara, CARACTERISTICAS])
X = pre.transform(df[CARACTERISTICAS])
assert not np.isnan(X).any()
joblib.dump(pre, RAIZ / "models" / "preprocessor.joblib")

# ---- Archivo procesado ----
salida = df[["id", "fecha_orig", "objetivo", "split_random", "split_temporal"] + CARACTERISTICAS]
ruta = RAIZ / "data" / "processed" / "pd.parquet"
salida.to_parquet(ruta, index=False, compression="zstd")

# ---- Verificaciones ----
excluidas = {v for v, (c, _) in VARIABLES.items() if c.startswith("excluir") or c == "objetivo"}
assert not (set(CARACTERISTICAS) & excluidas), set(CARACTERISTICAS) & excluidas

temporal_esperada = pd.read_csv(RAIZ / "results" / "tabla04_particion_temporal.csv", index_col=0)["registros"]
temporal = salida["split_temporal"].value_counts()
coincide = {"entrenamiento": int(temporal_esperada.iloc[0]), "validacion": int(temporal_esperada.iloc[1]),
            "prueba": int(temporal_esperada.iloc[2])}
assert {k: int(temporal[k]) for k in coincide} == coincide, (dict(temporal), coincide)

def resumen(col):
    g = salida.groupby(col)["objetivo"].agg(registros="count", malos="sum")
    g["tasa_malos_pct"] = (100 * g["malos"] / g["registros"]).round(2)
    g["pct_del_total"] = (100 * g["registros"] / len(salida)).round(2)
    return g.loc[["entrenamiento", "validacion", "prueba"]].reset_index().to_dict("records")

mb = ruta.stat().st_size / 1e6
assert mb < 100, f"el archivo pesa {mb:.1f} MB"
informe = {
    "registros_leidos": n_total, "registros_con_etiqueta": int(len(salida)),
    "caracteristicas_originales": len(USADAS), "columnas_numericas": len(NUMERICAS), "columnas_categoricas": len(CATEGORICAS),
    "columnas_de_salida_del_preprocesador": int(X.shape[1]),
    "semilla": SEMILLA, "proporcion_aleatoria": PROPORCION_ALEATORIA,
    "preprocesador_ajustado_con": AJUSTE_PREPROCESADOR, "registros_usados_para_ajustar": int(mascara.sum()),
    "particion_aleatoria": resumen("split_random"), "particion_temporal": resumen("split_temporal"),
    "valores_vacios_antes_de_imputar": {c: int(n) for c, n in salida[CARACTERISTICAS].isna().sum().items() if n > 0},
    "tamano_parquet_MB": round(mb, 2),
    "versiones": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                  "scikit-learn": sklearn.__version__, "pyarrow": pyarrow.__version__, "joblib": joblib.__version__},
}
(RAIZ / "results" / "verificacion_datos_procesados.json").write_text(
    json.dumps(informe, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
print(json.dumps(informe, indent=2, ensure_ascii=False, default=int))
