"""Codificación de la variable objetivo de la Parte 1 (PD), según la tabla del enunciado."""

# 0 = buen pagador, 1 = mal pagador, None = NA (se excluye del entrenamiento)
MAPA_PD = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Current": None,
    "Issued": None,
    "In Grace Period": None,
    "Late (16-30 days)": None,
}


def codificar_pd(loan_status):
    """Devuelve una serie con 0, 1 o NaN; las categorías no previstas quedan como NaN."""
    return loan_status.map(MAPA_PD)
