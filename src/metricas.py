"""Métricas de la Parte 1 (PD): discriminación y calibración.

Función común para todos los modelos de PD (regresión logística y red neuronal), de modo que las
métricas se calculen exactamente igual. Convención: y = 1 es el mal pagador (incumple) y p es la
probabilidad estimada de incumplimiento.

  AUC    = P(p_malo > p_bueno) + 0,5 · P(p_malo = p_bueno)            (área bajo la curva ROC)
  Gini   = 2 · AUC − 1
  KS     = max_t | F_malos(t) − F_buenos(t) |                          (F: distribución acumulada de p)
  Brier  = (1/n) Σ (p_i − y_i)²
  Log-loss = −(1/n) Σ [ y_i ln p_i + (1 − y_i) ln(1 − p_i) ]
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def metricas_pd(y, p):
    """Devuelve AUC, Gini, KS, Brier y log-loss para etiquetas y (0/1) y probabilidades p."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": float(np.max(tpr - fpr)),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
    }


def curva_calibracion(y, p, n_grupos=10):
    """Curva de calibración por grupos de igual tamaño (deciles de la probabilidad predicha)."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    d["grupo"] = pd.qcut(d["p"].rank(method="first"), n_grupos, labels=False)
    g = d.groupby("grupo").agg(n=("y", "size"), prob_predicha_media=("p", "mean"), tasa_observada=("y", "mean"))
    return g.reset_index()
