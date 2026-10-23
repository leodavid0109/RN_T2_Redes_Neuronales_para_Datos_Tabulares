"""Pruebas de la scorecard. Se ejecutan con:  python tests/test_scorecard.py   (o con pytest, si está instalado)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.scorecard import (Scorecard, factor_offset, probabilidad_desde_puntaje,  # noqa: E402
                           puntaje_desde_odds, puntaje_desde_probabilidad)

PDO, BASE, ODDS = 20, 600, 50


def test_duplicar_odds_suma_pdo():
    odds = np.array([0.05, 0.3, 1, 3.57, 10, 50, 400, 2000])
    dif = puntaje_desde_odds(2 * odds, PDO, BASE, ODDS) - puntaje_desde_odds(odds, PDO, BASE, ODDS)
    assert np.allclose(dif, PDO, rtol=0, atol=1e-9), dif


def test_puntaje_base_en_odds_de_referencia():
    assert abs(float(puntaje_desde_odds(ODDS, PDO, BASE, ODDS)) - BASE) < 1e-9


def test_factor_y_offset():
    factor, offset = factor_offset(PDO, BASE, ODDS)
    assert abs(factor - PDO / np.log(2)) < 1e-12
    assert abs(offset - (BASE - factor * np.log(ODDS))) < 1e-12


def test_ida_y_vuelta_y_monotonia():
    p = np.linspace(0.01, 0.99, 99)
    s = puntaje_desde_probabilidad(p, PDO, BASE, ODDS)
    assert np.allclose(probabilidad_desde_puntaje(s, PDO, BASE, ODDS), p, atol=1e-12)
    assert np.all(np.diff(s) < 0), "más probabilidad de incumplir debe dar menos puntos"


def test_contribuciones_suman_el_puntaje_y_faltantes_coinciden():
    if not (RAIZ / "models" / "scorecard_referencia.json").exists():
        print("(se omite: ejecute scripts/10_scorecard.py primero)")
        return
    sc = Scorecard.cargar(RAIZ)
    df = pd.read_parquet(RAIZ / "data" / "processed" / "pd.parquet").sample(2000, random_state=0)
    base = df[[c for c in sc.pre.feature_names_in_ if not c.startswith("falta_")]]
    # los indicadores de faltante se reconstruyen desde las 24 variables, igual que en el parquet
    assert (sc.preparar(base)[[c for c in sc.pre.feature_names_in_ if c.startswith("falta_")]].to_numpy()
            == df[[c for c in sc.pre.feature_names_in_ if c.startswith("falta_")]].to_numpy()).all()
    s = sc.puntaje(base)
    assert np.allclose(sc.puntaje_promedio + sc.contribuciones(base).sum(axis=1), s, atol=1e-8)
    assert np.allclose(s, puntaje_desde_probabilidad(sc.probabilidad(base), PDO, BASE, ODDS), atol=1e-8)
    m = sc.puntaje_mostrado(base)
    assert m.min() >= sc.puntaje_minimo and m.max() <= sc.puntaje_maximo
    dentro = (s >= sc.puntaje_minimo) & (s <= sc.puntaje_maximo)
    assert np.allclose(m[dentro], s[dentro])
    e = sc.escenario(base.iloc[0], {"annual_inc": 1e9})   # ingreso absurdo: el puntaje mostrado no se sale del rango
    assert e["puntaje_nuevo"] == sc.puntaje_maximo and 0 <= e["percentil_nuevo"] <= 100


if __name__ == "__main__":
    n = 0
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            n += 1
            print("OK ", nombre)
    print(f"{n} pruebas pasaron")
