"""Scorecard de la PD: transformación de probabilidad a puntaje, puntos por variable, percentil y escenarios.

Definiciones (odds = P(bueno) / P(malo) = (1 − PD) / PD):
    Factor  = PDO / ln 2
    Offset  = puntaje_base − Factor · ln(odds_base)
    Puntaje = Offset + Factor · ln(odds)
Con ello, duplicar las odds suma exactamente PDO puntos y odds = odds_base da puntaje_base.

Puntos desde la regresión logística (logit = b + Σ_j w_j x_j sobre las columnas del preprocesador):
    ln(odds) = −logit    →    Puntaje = (Offset − Factor·b) + Σ_j (−Factor · w_j · x_j)
Cada columna aporta puntos aditivos; se agrupan por variable original y se centran en la población de
referencia, de modo que la contribución de una variable es 0 para el solicitante promedio:
    Puntaje = Puntaje_promedio + Σ_v ( C_v(x) − media_v ),   C_v(x) = −Factor · Σ_{j ∈ v} w_j x_j
Mayor puntaje = menor riesgo. Para mostrar al usuario el puntaje se acota a [puntaje_minimo, puntaje_maximo]
(configs/scorecard.yaml); el puntaje sin acotar se usa para evaluar y para los percentiles.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.variables import CATEGORICAS, INDICADOR_FALTANTE

RAIZ = Path(__file__).resolve().parents[1]
ORIGEN_COLUMNA = {"term_meses": "term", "emp_length_anios": "emp_length", "antiguedad_cred_meses": "earliest_cr_line"}


def factor_offset(pdo, puntaje_base, odds_base):
    factor = pdo / np.log(2)
    return factor, puntaje_base - factor * np.log(odds_base)


def puntaje_desde_odds(odds, pdo, puntaje_base, odds_base):
    factor, offset = factor_offset(pdo, puntaje_base, odds_base)
    return offset + factor * np.log(np.asarray(odds, dtype=float))


def puntaje_desde_probabilidad(pd_incumple, pdo, puntaje_base, odds_base):
    p = np.clip(np.asarray(pd_incumple, dtype=float), 1e-300, 1 - 1e-16)
    return puntaje_desde_odds((1 - p) / p, pdo, puntaje_base, odds_base)


def probabilidad_desde_puntaje(puntaje, pdo, puntaje_base, odds_base):
    factor, offset = factor_offset(pdo, puntaje_base, odds_base)
    odds = np.exp((np.asarray(puntaje, dtype=float) - offset) / factor)
    return 1.0 / (1.0 + odds)


def variable_original(columna):
    """Variable de las 24 originales de la que proviene una columna del preprocesador."""
    if columna.startswith("falta_"):
        return columna[len("falta_"):]
    if columna in ORIGEN_COLUMNA:
        return ORIGEN_COLUMNA[columna]
    for c in CATEGORICAS:
        if columna.startswith(c + "_"):
            return c
    return columna


class Scorecard:
    def __init__(self, pre, modelo, pdo, puntaje_base, odds_base, medias, puntajes_referencia,
                 puntaje_minimo=-np.inf, puntaje_maximo=np.inf):
        self.pre, self.modelo = pre, modelo
        self.puntaje_minimo, self.puntaje_maximo = puntaje_minimo, puntaje_maximo
        self.pdo, self.puntaje_base, self.odds_base = pdo, puntaje_base, odds_base
        self.factor, self.offset = factor_offset(pdo, puntaje_base, odds_base)
        self.columnas = list(pre.get_feature_names_out())
        self.variables_de_columna = [variable_original(c) for c in self.columnas]
        self.variables = list(dict.fromkeys(self.variables_de_columna))
        self.w = modelo.coef_[0]
        self.b = float(modelo.intercept_[0])
        assert len(self.w) == len(self.columnas)
        self.medias = {v: float(medias[v]) for v in self.variables}
        self.puntaje_promedio = self.offset - self.factor * self.b + sum(self.medias.values())
        self.referencia = np.sort(np.asarray(puntajes_referencia, dtype=float))

    # ---------- construcción y carga ----------
    @staticmethod
    def _config(raiz):
        return yaml.safe_load((Path(raiz) / "configs" / "scorecard.yaml").read_text(encoding="utf-8"))["scorecard"]

    @classmethod
    def construir(cls, df_referencia, raiz=RAIZ):
        """Calcula las medias por variable con la población de referencia (DataFrame con las características)."""
        cfg = cls._config(raiz)
        pre = joblib.load(Path(raiz) / cfg["preprocesador"])
        modelo = joblib.load(Path(raiz) / cfg["modelo"])
        lim = {"puntaje_minimo": cfg.get("puntaje_minimo", -np.inf), "puntaje_maximo": cfg.get("puntaje_maximo", np.inf)}
        tmp = cls(pre, modelo, cfg["pdo"], cfg["puntaje_base"], cfg["odds_base"],
                  {v: 0.0 for v in dict.fromkeys(variable_original(c) for c in pre.get_feature_names_out())}, [0.0], **lim)
        medias = tmp._contribuciones_brutas(df_referencia).mean()
        sc = cls(pre, modelo, cfg["pdo"], cfg["puntaje_base"], cfg["odds_base"], medias.to_dict(), [0.0], **lim)
        sc.referencia = np.sort(sc.puntaje(df_referencia))
        return sc

    def guardar(self, raiz=RAIZ):
        (Path(raiz) / "models" / "scorecard_referencia.json").write_text(
            json.dumps({"medias_contribucion": self.medias, "puntaje_promedio": self.puntaje_promedio,
                        "n_referencia": int(len(self.referencia))}, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def cargar(cls, raiz=RAIZ):
        """Carga la scorecard lista para usar (la app). Percentiles desde results/score_distribution.csv."""
        cfg = cls._config(raiz)
        info = json.loads((Path(raiz) / "models" / "scorecard_referencia.json").read_text(encoding="utf-8"))
        dist = pd.read_csv(Path(raiz) / "results" / "score_distribution.csv")
        return cls(joblib.load(Path(raiz) / cfg["preprocesador"]), joblib.load(Path(raiz) / cfg["modelo"]),
                   cfg["pdo"], cfg["puntaje_base"], cfg["odds_base"], info["medias_contribucion"], dist["puntaje"].to_numpy(),
                   puntaje_minimo=cfg.get("puntaje_minimo", -np.inf), puntaje_maximo=cfg.get("puntaje_maximo", np.inf))

    # ---------- cálculo ----------
    def preparar(self, df):
        """Acepta las 24 variables en su forma procesada y agrega los indicadores de faltante."""
        df = df.copy()
        for v in INDICADOR_FALTANTE:
            df[f"falta_{v}"] = df[v].isna().astype(int)
        return df[list(self.pre.feature_names_in_)]

    def _X(self, df):
        return self.pre.transform(self.preparar(df))

    def probabilidad(self, df):
        return self.modelo.predict_proba(self._X(df))[:, 1]

    def puntaje(self, df):
        return self.offset - self.factor * (self._X(df) @ self.w + self.b)

    def puntaje_mostrado(self, df):
        """Puntaje acotado al rango que se muestra al usuario."""
        return np.clip(self.puntaje(df), self.puntaje_minimo, self.puntaje_maximo)

    def _contribuciones_brutas(self, df):
        puntos = -self.factor * self._X(df) * self.w                      # n × columnas
        return pd.DataFrame(puntos, columns=self.columnas).T.groupby(self.variables_de_columna, sort=False).sum().T

    def contribuciones(self, df):
        """Puntos de cada variable respecto del solicitante promedio (suman, junto con el promedio, el puntaje)."""
        c = self._contribuciones_brutas(df)
        return c - pd.Series(self.medias)[c.columns]

    def percentil(self, puntaje):
        """Porcentaje de la población de referencia con puntaje menor o igual (más alto = mejor que más gente)."""
        return 100.0 * np.searchsorted(self.referencia, np.asarray(puntaje, dtype=float), side="right") / len(self.referencia)

    def escenario(self, solicitante, cambios):
        """Compara el puntaje de un solicitante antes y después de cambiar algunas características."""
        base = pd.DataFrame([dict(solicitante)])
        nuevo = base.copy()
        for k, v in cambios.items():
            nuevo[k] = v
        s0, s1 = float(self.puntaje(base)[0]), float(self.puntaje(nuevo)[0])
        m0, m1 = (float(np.clip(s, self.puntaje_minimo, self.puntaje_maximo)) for s in (s0, s1))
        return {"puntaje_actual": m0, "puntaje_nuevo": m1, "cambio": m1 - m0,
                "pd_actual": float(self.probabilidad(base)[0]), "pd_nueva": float(self.probabilidad(nuevo)[0]),
                "percentil_actual": float(self.percentil(s0)), "percentil_nuevo": float(self.percentil(s1))}
