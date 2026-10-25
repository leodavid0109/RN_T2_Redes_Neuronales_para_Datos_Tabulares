"""Métricas de la Parte 2 (LGD): regresión sobre una fracción en [0, 1].

Función común para todos los modelos de LGD (lineal y red neuronal). Con y la LGD observada
y p la predicha (recortada a [0, 1]):

  MAE   = (1/n) Σ |p_i − y_i|
  RMSE  = sqrt( (1/n) Σ (p_i − y_i)² )
  Sesgo = (1/n) Σ (p_i − y_i)            (negativo: el modelo subestima la LGD)
  R²    = 1 − Σ (y_i − p_i)² / Σ (y_i − ȳ)²
"""
import numpy as np


def metricas_lgd(y, p, recortar=True):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if recortar:
        p = np.clip(p, 0.0, 1.0)
    e = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "sesgo": float(np.mean(e)),
        "r2": float(1 - np.sum(e ** 2) / ss_tot),
    }