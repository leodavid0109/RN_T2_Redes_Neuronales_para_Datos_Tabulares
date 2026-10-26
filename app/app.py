"""App web de la scorecard: puntaje, probabilidad de incumplimiento, percentil y escenarios.

Uso (desde la raiz del repo):
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.scorecard import Scorecard  # noqa: E402
from src.variables import INDICADOR_FALTANTE  # noqa: E402

URL_REPORTE = "https://PEGAR-ENLACE-DEL-BLOG-UWU"
URL_VIDEO = "https://PEGAR-ENLACE-DEL-VIDEO-UWU"

# Etiquetas en lenguaje claro. Verificar cada una contra el diccionario de variables del conjunto.
ETIQUETAS = {
    "loan_amnt": "Monto del préstamo (USD)",
    "term_meses": "Plazo (meses)",
    "emp_length_anios": "Antigüedad en el empleo (años)",
    "annual_inc": "Ingreso anual (USD)",
    "dti": "Relación deuda/ingreso, DTI (%)",
    "delinq_2yrs": "Atrasos de más de 30 días en los últimos 2 años",
    "antiguedad_cred_meses": "Antigüedad del historial crediticio (meses)",
    "inq_last_6mths": "Consultas de crédito en los últimos 6 meses",
    "open_acc": "Líneas de crédito abiertas",
    "pub_rec": "Registros públicos negativos",
    "revol_bal": "Saldo de crédito rotativo (USD)",
    "revol_util": "Uso del crédito rotativo (%)",
    "total_acc": "Total de líneas de crédito (históricas)",
    "home_ownership": "Tipo de vivienda",
    "verification_status": "Verificación de ingresos",
    "purpose": "Propósito del préstamo",
    "addr_state": "Estado (EE. UU.)",
}
PRINCIPALES = ["loan_amnt", "term_meses", "annual_inc", "dti", "emp_length_anios", "home_ownership",
               "purpose", "verification_status", "revol_util", "inq_last_6mths", "delinq_2yrs",
               "antiguedad_cred_meses"]

st.set_page_config(page_title="¿Me prestas?", page_icon="💳", layout="wide")


@st.cache_resource
def cargar():
    sc = Scorecard.cargar()
    datos = json.loads((RAIZ / "app/datos_app.json").read_text(encoding="utf-8"))
    return sc, datos


def campo(col, meta, clave, valor=None):
    """Widget de una variable, con rangos válidos tomados de los datos."""
    etiqueta = ETIQUETAS.get(col, col)
    if meta["tipo"] == "cat":
        op = meta["valores"]
        i = op.index(valor) if valor in op else op.index(meta["defecto"])
        return st.selectbox(etiqueta, op, index=i, key=clave)
    ref = meta["defecto"] if valor is None else float(valor)
    if meta["min"] == meta["max"]:
        return meta["min"]
    if meta["valores"]:
        op = [float(v) for v in meta["valores"]]
        i = int(np.argmin([abs(v - ref) for v in op]))
        return st.selectbox(etiqueta, op, index=i, key=clave, format_func=lambda v: f"{v:g}")
    ref = float(min(max(ref, meta["min"]), meta["max"]))
    return st.number_input(etiqueta, min_value=meta["min"], max_value=meta["max"], value=ref,
                           step=1.0 if meta["entero"] else 0.1,
                           format="%.0f" if meta["entero"] else "%.1f", key=clave,
                           help=f"Rango observado: {meta['min']:,.0f} a {meta['max']:,.0f}")


def leer(col, meta, prefijo):
    if col in INDICADOR_FALTANTE:
        sin = st.checkbox(f"{ETIQUETAS.get(col, col)}: sin registro / no aplica",
                          value=meta["falta"] > 0.5, key=f"{prefijo}_sin_{col}")
        if sin:
            return np.nan
    return campo(col, meta, f"{prefijo}_{col}")


sc, datos = cargar()

st.title("💳 ¿Me prestas? Tu puntaje de crédito")
st.caption("Ejercicio académico (Universidad Nacional de Colombia Sede Medellín, Facultad de Minas): no es una decisión de "
           "crédito real. Los datos que ingreses no se guardan.")
st.markdown(f"[Reporte técnico]({URL_REPORTE})  |  [Video]({URL_VIDEO})")

izq, der = st.columns([1.1, 1])

with izq:
    st.subheader("1. Cuéntanos de ti")
    st.caption("Ya hay un ejemplo cargado. Cámbialo con tus datos.")
    sol = {}
    principales = [c for c in PRINCIPALES if c in datos]
    a, b = st.columns(2)
    for i, c in enumerate(principales):
        with (a if i % 2 == 0 else b):
            sol[c] = leer(c, datos[c], "a")
    with st.expander("Más variables (opcional)"):
        for c in datos:
            if c not in sol:
                sol[c] = leer(c, datos[c], "a")

df = pd.DataFrame([{c: sol[c] for c in datos}])
puntaje_crudo = float(sc.puntaje(df)[0])
puntaje = float(sc.puntaje_mostrado(df)[0])
prob = float(sc.probabilidad(df)[0])
pct = float(sc.percentil(puntaje_crudo))

with der:
    st.subheader("2. Tu resultado")
    m1, m2, m3 = st.columns(3)
    m1.metric("Puntaje", f"{puntaje:.0f}")
    m2.metric("Prob. de incumplimiento", f"{prob:.1%}")
    m3.metric("Percentil", f"{pct:.0f}")
    st.write(f"Tu puntaje es igual o mayor que el **{pct:.0f} %** de la población de referencia. "
             "A mayor puntaje, menor riesgo.")
    st.caption(f"Escala: {sc.puntaje_base:g} puntos equivalen a odds de {sc.odds_base:g} (buenos por cada malo); "
               f"cada {sc.pdo:g} puntos las odds se duplican.")
    conteo, bordes = np.histogram(sc.referencia, bins=40)
    centros = np.round((bordes[:-1] + bordes[1:]) / 2).astype(int)
    st.caption("Distribución de puntajes de la población de referencia")
    st.bar_chart(pd.Series(conteo, index=centros))
    st.caption(f"Tu puntaje: {puntaje_crudo:.0f}")
    st.caption("Qué sube o baja tu puntaje frente al solicitante promedio (puntos por variable)")
    contrib = sc.contribuciones(df).iloc[0].rename(index=lambda c: ETIQUETAS.get(c, c))
    st.bar_chart(contrib.sort_values())

st.divider()
st.subheader("3. ¿Y si cambiara…?")
opciones = [c for c in principales if c not in INDICADOR_FALTANTE]
por_defecto = [c for c in ["dti"] if c in opciones]
elegidas = st.multiselect("Elige qué característica cambiar", opciones, default=por_defecto,
                          format_func=lambda c: ETIQUETAS.get(c, c))
cambios = {}
cols_esc = st.columns(max(len(elegidas), 1))
for col_ui, c in zip(cols_esc, elegidas):
    with col_ui:
        # la clave incluye el valor base: si cambias el formulario, el escenario se reinicia
        cambios[c] = campo(c, datos[c], f"e_{c}_{sol[c]}", valor=sol[c])
if cambios:
    r = sc.escenario(sol, cambios)
    e1, e2, e3 = st.columns(3)
    e1.metric("Puntaje nuevo", f"{r['puntaje_nuevo']:.0f}", f"{r['cambio']:+.0f} puntos")
    e2.metric("Prob. de incumplimiento", f"{r['pd_nueva']:.1%}",
              f"{(r['pd_nueva'] - r['pd_actual']) * 100:+.1f} pp", delta_color="inverse")
    e3.metric("Percentil", f"{r['percentil_nuevo']:.0f}", f"{r['percentil_nuevo'] - r['percentil_actual']:+.0f}")
