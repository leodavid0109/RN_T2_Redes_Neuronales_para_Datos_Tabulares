"""Propuesta de clasificación de las 74 columnas del CSV para el modelo de PD (Parte 1).

ES UNA PROPUESTA para revisar y firmar: cada decisión se puede cambiar editando este archivo.
Supuesto de caso de uso: quien usa el modelo evalúa una solicitud ANTES de otorgar el crédito,
con la información de la solicitud y del historial crediticio disponible en ese momento.
El diccionario no indica con precisión en qué momento se captura cada variable; para las
variables del historial crediticio se asume que corresponden al momento de la solicitud.

Categorías:
  usar               -> entra al modelo
  excluir_ligada_lc  -> ligada a la calificación de Lending Club o ajena al solicitante
  excluir_posterior  -> ocurre después del otorgamiento (fuga de información)
  excluir_identificador, excluir_texto, excluir_faltantes, excluir_constante, excluir_redundante
  objetivo / auxiliar
"""

VARIABLES = {
    # ---- Objetivo y auxiliares ----
    "loan_status": ("objetivo", "Origen de la variable objetivo; no es característica."),
    "issue_d": ("auxiliar", "Fecha de originación: solo se usa para la partición temporal, no como característica."),

    # ---- Usar ----
    "loan_amnt": ("usar", "Monto solicitado; se conoce al solicitar el crédito."),
    "term": ("usar", "Plazo (36 o 60 meses); se pacta al otorgar."),
    "emp_length": ("usar", "Antigüedad laboral declarada en la solicitud (3,99 % faltantes)."),
    "home_ownership": ("usar", "Situación de vivienda declarada en el registro."),
    "annual_inc": ("usar", "Ingreso anual autodeclarado en el registro."),
    "verification_status": ("usar", "Verificación del ingreso por LC al evaluar la solicitud. No está en el diccionario con este nombre; equivale probablemente a is_inc_v (a confirmar)."),
    "purpose": ("usar", "Categoría del crédito indicada por el prestatario en la solicitud."),
    "dti": ("usar", "Razón deuda/ingreso calculada con la información de la solicitud."),
    "delinq_2yrs": ("usar", "Historial crediticio: moras de 30+ días en los 2 años previos."),
    "earliest_cr_line": ("usar", "Se transforma en antigüedad crediticia (meses entre earliest_cr_line e issue_d)."),
    "inq_last_6mths": ("usar", "Consultas de crédito en los 6 meses previos."),
    "open_acc": ("usar", "Líneas de crédito abiertas en el historial."),
    "pub_rec": ("usar", "Registros públicos negativos."),
    "revol_bal": ("usar", "Saldo rotativo total."),
    "revol_util": ("usar", "Tasa de utilización de la línea rotativa (0,09 % faltantes)."),
    "total_acc": ("usar", "Total de líneas de crédito en el historial."),
    "acc_now_delinq": ("usar", "Cuentas en mora al momento de la solicitud."),
    "collections_12_mths_ex_med": ("usar", "Cobranzas (sin médicas) en los últimos 12 meses."),
    "mths_since_last_delinq": ("usar", "55 % faltante: el vacío significa probablemente que nunca hubo mora. Se imputa y se agrega un indicador de faltante."),
    "mths_since_last_record": ("usar", "87 % faltante: el vacío significa probablemente que no hay registros públicos. Se imputa y se agrega un indicador de faltante."),

    # ---- Decidir ----
    "grade": ("excluir_ligada_lc", "Calificación asignada por LC; int_rate e installment dependen de ella, por lo que usarla haría que el modelo reprodujera la evaluación de la plataforma."),
    "sub_grade": ("excluir_ligada_lc", "Subcalificación asignada por LC; mismo motivo que grade."),
    "int_rate": ("excluir_ligada_lc", "Aumenta sistemáticamente con grade (tasa media de 7,2 % en el grado A y de 25,6 % en el G)."),
    "installment": ("excluir_ligada_lc", "Coincide con la cuota calculada con monto, tasa y plazo en el 99,5 % de los créditos, por lo que arrastra int_rate."),
    "addr_state": ("usar", "51 categorías; puede actuar como sustituto de otras características, por lo que su efecto se discute con los resultados."),
    "initial_list_status": ("excluir_ligada_lc", "Estado inicial de publicación del crédito (valores W y F); no describe al solicitante."),
    "tot_coll_amt": ("usar", "No existe antes de ago-2012 (24,8 % de faltantes estructurales); se imputa y se agrega indicador de faltante."),
    "tot_cur_bal": ("usar", "No existe antes de ago-2012 (24,8 % de faltantes estructurales); se imputa y se agrega indicador de faltante."),
    "total_rev_hi_lim": ("usar", "No existe antes de ago-2012 (24,8 % de faltantes estructurales); se imputa y se agrega indicador de faltante."),
    "mths_since_last_major_derog": ("excluir_faltantes", "80,9 % de faltantes y solo existe desde ago-2012."),

    # ---- Excluir: posteriores al otorgamiento (fuga de información) ----
    "out_prncp": ("excluir_posterior", "Capital pendiente: solo existe una vez que el crédito empieza a pagarse."),
    "out_prncp_inv": ("excluir_posterior", "Capital pendiente de la porción de inversionistas: posterior al otorgamiento."),
    "total_pymnt": ("excluir_posterior", "Pagos recibidos a la fecha: posterior al otorgamiento."),
    "total_pymnt_inv": ("excluir_posterior", "Pagos recibidos de la porción de inversionistas: posterior al otorgamiento."),
    "total_rec_prncp": ("excluir_posterior", "Capital recibido a la fecha: posterior al otorgamiento."),
    "total_rec_int": ("excluir_posterior", "Intereses recibidos a la fecha: posterior al otorgamiento."),
    "total_rec_late_fee": ("excluir_posterior", "Cargos por mora recibidos: delatan directamente el comportamiento de pago."),
    "recoveries": ("excluir_posterior", "Recuperación posterior al castigo: solo existe si el crédito incumplió. Se usa para construir la LGD, no como característica."),
    "collection_recovery_fee": ("excluir_posterior", "Comisión de cobranza posterior al castigo. Se usa para construir la LGD, no como característica."),
    "last_pymnt_d": ("excluir_posterior", "Fecha del último pago recibido: posterior al otorgamiento."),
    "last_pymnt_amnt": ("excluir_posterior", "Monto del último pago recibido: posterior al otorgamiento."),
    "next_pymnt_d": ("excluir_posterior", "Próximo pago programado: posterior al otorgamiento (94 % faltante)."),
    "last_credit_pull_d": ("excluir_posterior", "Última consulta de crédito de LC sobre este crédito, durante su vida."),
    "pymnt_plan": ("excluir_posterior", "Plan de pagos puesto en marcha durante la vida del crédito; además es 'n' en casi todos los registros."),

    # ---- Excluir: identificadores ----
    "id": ("excluir_identificador", "Identificador del crédito; no tiene poder predictivo."),
    "member_id": ("excluir_identificador", "Identificador del prestatario; no tiene poder predictivo."),
    "url": ("excluir_identificador", "Enlace construido con el id del crédito."),

    # ---- Excluir: texto libre ----
    "emp_title": ("excluir_texto", "Texto libre con 138.599 valores distintos; requeriría procesamiento de lenguaje."),
    "desc": ("excluir_texto", "Texto libre del prestatario con 65,75 % faltante."),
    "title": ("excluir_texto", "Texto libre (49.883 valores) que en gran parte repite a purpose."),

    # ---- Excluir: redundantes ----
    "zip_code": ("excluir_redundante", "890 categorías; la ubicación ya está en addr_state."),
    "funded_amnt": ("excluir_redundante", "Monto comprometido tras la aprobación y muy cercano a loan_amnt. Se conserva fuera del modelo: sirve para el EAD de la Parte 2."),
    "funded_amnt_inv": ("excluir_redundante", "Monto comprometido por inversionistas, definido tras la aprobación."),

    # ---- Excluir: constantes ----
    "policy_code": ("excluir_constante", "Tiene un único valor en los registros etiquetados."),
    "application_type": ("excluir_constante", "Prácticamente todos los registros son INDIVIDUAL."),

    # ---- Excluir: casi sin datos (aparecen solo a partir de fines de 2015) ----
    "annual_inc_joint": ("excluir_faltantes", "Solo 3 de 268.530 registros etiquetados tienen dato."),
    "dti_joint": ("excluir_faltantes", "Solo 3 de 268.530 registros etiquetados tienen dato."),
    "verification_status_joint": ("excluir_faltantes", "Solo 3 de 268.530 registros etiquetados tienen dato. Sin definición en el diccionario con este nombre."),
    "open_acc_6m": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "open_il_6m": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "open_il_12m": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "open_il_24m": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "mths_since_rcnt_il": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "total_bal_il": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "il_util": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "open_rv_12m": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "open_rv_24m": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "max_bal_bc": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "all_util": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "inq_fi": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "total_cu_tl": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
    "inq_last_12m": ("excluir_faltantes", "99,95 % faltante; existe solo desde dic-2015."),
}

# El CSV usa estos nombres; el diccionario usa otros. La equivalencia es una suposición a confirmar.
NOMBRE_EN_DICCIONARIO = {
    "verification_status": "is_inc_v",
    "verification_status_joint": "verified_status_joint",
}


# ---- Listas derivadas (decisión final del equipo) ----
USADAS = [v for v, (c, _) in VARIABLES.items() if c == "usar"]
CATEGORICAS = ["home_ownership", "verification_status", "purpose", "addr_state"]
INDICADOR_FALTANTE = ["mths_since_last_delinq", "mths_since_last_record",
                      "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim"]
