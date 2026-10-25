# LGD: auditoría del Ridge y arranque del MLP de Persona B

## Alcance y estado

Este trabajo añade un MLP demo separado; no modifica los artefactos, scripts ni resultados de C. El modelo principal existente sigue siendo Ridge en `models/lgd_lineal_{aleatoria,temporal}.joblib`. La salida del MLP vive en `models/lgd_mlp_demo/` y `results/lgd_mlp_demo/`; no se enchufa aún a la pérdida esperada.

La ejecución usa el preprocesador compartido actual. Los metadatos guardan `preprocesador_ajustado_con: entrenamiento_aleatorio`; puede haber transformado filas que pertenecen a validación/prueba temporal. Este MLP temporal es provisional por la misma limitación conocida en PD. La decisión del equipo de conservar el preprocesador no demuestra que esté libre de fuga.

## Mapa del flujo auditado

```text
data/processed/pd.parquet (PD=1 + máscaras y covariables)
       │ src/lgd.py cruza por id con el CSV original (no presente localmente)
       ├─ funded_amnt, total_rec_prncp → EAD = funded_amnt − total_rec_prncp
       ├─ recoveries − collection_recovery_fee → LGD = clip(1 − recupero_neto/EAD, 0, 1)
       └─ estado del préstamo → castigado
                    ↓
data/processed/lgd.parquet (58.814 filas PD=1, EAD>0 y cruce válido)
       ├─ castigado=True (46.009): muestra objetivo de Ridge y MLP
       └─ máscaras split_random / split_temporal
                    ↓ preprocessor.joblib compartido: 29 columnas → 99 features
       ├─ src/lgd_lineal.py: Ridge, selección de alpha por RMSE de validación
       │     ├─ models/lgd_lineal_*.joblib
       │     ├─ results/lgd_lineal.csv (Ridge y media de entrenamiento)
       │     └─ results/lgd_lineal_pred.parquet (predicciones por id/partición/conjunto)
       ├─ scripts/16_entrenar_mlp_lgd.py (demo nueva, validación únicamente)
       └─ src/perdida_esperada.py: PD × LGD × funded_amnt, tabla por deciles y parquet por crédito
```

## Datos, objetivo y elegibilidad

El diccionario `data/raw/LCDataDictionary.xlsx` describe `id` como identificador único de LendingClub; `loan_status` como estado actual; `funded_amnt` como importe comprometido; `total_rec_prncp` como principal recibido hasta la fecha; `recoveries` como recuperación bruta posterior al charge-off y `collection_recovery_fee` como comisión posterior al charge-off. La palabra “actual” en `loan_status` y “to date/post charge off” en los otros campos importa: son observaciones posteriores al otorgamiento y pueden depender de cuánto tiempo ha tenido el préstamo para madurar.

El código define `EAD = funded_amnt − total_rec_prncp`, interpreta el resultado como principal aún pendiente al incumplir y elimina EAD no positivo. Define `LGD = clip(1 − (recoveries − collection_recovery_fee)/EAD, 0, 1)`. Así, una recuperación neta nula o negativa da LGD=1; recuperación neta igual a EAD da 0; límites protegen de resultados fuera del rango. La fórmula no incluye intereses y usa cantidades acumuladas del archivo original; no demuestra por sí sola que todas las recuperaciones estén medidas al mismo horizonte.

**Límite de verificación:** `data/raw/loan.csv` no está presente en este checkout, y el parquet procesado ya descartó `total_rec_prncp`, `recoveries` y `collection_recovery_fee`. Por tanto, verifiqué código, definiciones del diccionario, rango y distribución de los campos derivados, pero no pude recalcular cada LGD/EAD directamente desde las columnas originales. El archivo original haría falta para reproducir la fórmula registro a registro.

El parquet tiene **58.814** filas y todas tienen `objetivo=1`; el parquet PD tiene 58.819 incumplidos únicos. Cinco incumplidos no están en LGD por el cruce/filtrado de EAD, pero sin el CSV no se puede determinar cuántos fallaron cada paso. En el parquet LGD hay 46.009 castigados y 12.805 no castigados:

| Estado en LGD | Filas | Significado operativo en el código | LGD exactamente 1 |
|---|---:|---|---:|
| `Charged Off` | 45.248 | castigado=True | incluido en 21.332 LGD=1 del grupo castigado |
| `Does not meet the credit policy. Status:Charged Off` | 761 | castigado=True | incluido en ese mismo total |
| `Default` | 1.218 | castigado=False | 1.218 |
| `Late (31-120 days)` | 11.587 | castigado=False | 11.587 |

La fuente construye LGD para todos los `objetivo=1`, pero Ridge filtra `castigado=True`. Los 12.805 no castigados tienen LGD=1 en el parquet: es compatible con que aún no se haya observado el periodo de recuperaciones post charge-off, por lo que ese 1 puede ser censura y no severidad final. Ridge evita entrenar directamente contra esa masa de resultados incompletos. La futura red debe usar exactamente el mismo filtro para que la comparación sea justa. Al aplicarla a otros incumplidos, su salida será una extrapolación desde la población castigada, no una estimación validada con etiquetas maduras de los 12.805.

En total, LGD tiene media 0,944949, desviación 0,091655, 34.137 valores exactamente 1 (58,04 %) y 42 exactamente 0 (0,071 %). En la muestra castigada que realmente modela Ridge, la media es 0,929627, desviación 0,098287, 21.332 exactamente 1 (46,36 %) y 42 exactamente 0. Los objetivos son así continuos en [0,1] con un átomo grande en 1; no siguen una Beta ordinaria sin modificaciones porque Beta no admite masa exacta en los extremos.

### Muestra por partición

Las filas se heredan de las columnas `split_random` y `split_temporal` del parquet PD. La tabla muestra `todas` las LGD y, por separado, las que Ridge/MLP pueden entrenar (`castigado`). Cada incumplido ya tiene PD=1; la partición no vuelve a crear ese objetivo.

| Partición | Conjunto | LGD total | castigado (Ridge/MLP) | no castigado | castigados con LGD=1 | castigados con LGD=0 |
|---|---|---:|---:|---:|---:|---:|
| Aleatoria | entrenamiento | 41.171 | 32.168 | 9.003 | 14.934 | 31 |
| Aleatoria | validación | 8.821 | 6.913 | 1.908 | 3.177 | 4 |
| Aleatoria | prueba | 8.822 | 6.928 | 1.894 | 3.221 | 7 |
| Temporal | entrenamiento | 31.527 | 29.255 | 2.272 | 10.096 | 27 |
| Temporal | validación | 19.438 | 13.981 | 5.457 | 8.769 | 9 |
| Temporal | prueba | 7.849 | 2.773 | 5.076 | 2.467 | 6 |

El test temporal de la muestra elegible es pequeño y está compuesto mayoritariamente por incumplidos no castigados: solo 2.773 de 7.849 filas entran al modelo. Las cinco filas PD=1 que no aparecen en LGD se distribuyen como 1 en prueba aleatoria y 2 en prueba temporal (las otras dos están en los otros conjuntos). En `src/perdida_esperada.py` las filas sin LGD se convierten a pérdida realizada cero, mientras que el indicador de censura cuenta todo PD=1 con `castigado != True`. Por eso en temporal prueba el resumen informa 5.078 censurados, aunque el parquet contiene 5.076 no castigados: se agregan dos filas ausentes de LGD. La cota superior de pérdida realizada no cubre correctamente esos dos casos. No modifiqué el código de C; conviene corregir/conciliar este caso antes de defender esos límites de pérdida.

## Predictores, preprocesamiento y fuga temporal

El artefacto compartido es un `ColumnTransformer`: imputa numéricas con mediana, las estandariza y codifica categorías one-hot densas con categorías desconocidas ignoradas. Tiene 29 columnas de entrada y produce 99 columnas. Verifiqué que ninguna de ellas sea `objetivo`, `lgd`, `ead`, `funded_amnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`, `castigado`, `loan_status`, `id`, `fecha_orig` ni las máscaras. Los 29 predictores son variables del préstamo/solicitante disponibles para la PD (montos, plazo, ingreso, DTI, historial, vivienda, verificación, propósito y estado geográfico), incluidos cinco indicadores de ausencia.

`src/lgd_lineal.py` filtra castigados y llama `pre.transform(df)`; no ajusta de nuevo. Comprobé que el MLP produce una matriz densa finita `(46.009, 99)` con las mismas columnas, y que los conjuntos de IDs de entrenamiento, validación y prueba de ambos protocolos coinciden exactamente con `results/lgd_lineal_pred.parquet`. Para ser comparable, el MLP reutiliza ese filtro, el orden de variables, el preprocesador y las máscaras; no debe volver a ajustar escalado/imputación.

El preprocesador se ajustó con entrenamiento aleatorio. Como las filas de temporal se solapan con las particiones aleatorias, su ajuste pudo incluir filas temporales de validación/prueba. La evaluación temporal de Ridge y MLP mantiene esta limitación metodológica hasta que el equipo decida/regenera otro preprocesador; no se interpreta como ausencia de fuga.

## Ridge y referencia de media

`src/lgd_lineal.py` no tiene archivo de configuración LGD: acepta `--data`, `--pre`, `--alphas`, `--seed` y `--demo`; por defecto sus alphas son `[0.01, 0.1, 1, 10, 100, 1000]`. Sin `--demo`, conserva la muestra completa. Ajusta un Ridge sobre entrenamiento, selecciona alpha con menor RMSE de validación y no reajusta sobre train+validación. Evalúa los tres conjuntos después. La referencia `media_train` predice a todos la media LGD del entrenamiento.

Los artefactos guardados eligen alpha=100 aleatoria y alpha=100.000 temporal, ambos con 100 parámetros (99 coeficientes e intercepto). La búsqueda es validación-only, pero `results/lgd_lineal_resumen.json` solo registra ocho configuraciones y 0,98 s totales; no guarda la lista exacta. El valor temporal 100.000 excede el máximo 1.000 del valor por defecto del código. Esto implica que se ejecutó con una lista explícita diferente o que código/resultados no corresponden; la lista usada no puede reconstruirse de los artefactos actuales. El CSV `lgd_lineal.csv` contiene el alpha elegido, métricas por conjunto y tiempo de búsqueda, pero no todas las filas/candidatos de la búsqueda. Es una brecha de reproducibilidad pendiente.

| Partición / conjunto | Modelo | MAE | RMSE | Sesgo (pred−obs) | R² |
|---|---|---:|---:|---:|---:|
| Aleatoria / validación | Media train | 0,073470 | 0,095436 | −0,000993 | −0,000108 |
| Aleatoria / validación | Ridge α=100 | 0,072832 | 0,094746 | −0,000865 | 0,014293 |
| Aleatoria / prueba | Media train | 0,072542 | 0,093810 | −0,003024 | −0,001040 |
| Aleatoria / prueba | Ridge α=100 | 0,071973 | 0,093120 | −0,003157 | 0,013630 |
| Temporal / validación | Media train | 0,077350 | **0,085752** | −0,027788 | **−0,117326** |
| Temporal / validación | Ridge α=100.000 | 0,077379 | 0,085859 | −0,028311 | −0,120107 |
| Temporal / prueba | Media train | **0,083958** | **0,094929** | −0,063384 | **−0,804489** |
| Temporal / prueba | Ridge α=100.000 | 0,084295 | 0,095254 | −0,063850 | −0,816891 |

En aleatoria, Ridge gana a la media por márgenes pequeños. En temporal, la media es ligeramente mejor tanto en validación como en prueba para MAE/RMSE/R². En ambas, el sesgo temporal negativo indica subestimación media, fuerte en test. La referencia aporta una base útil: demuestra que complejidad lineal no siempre mejora bajo cambio temporal y fija un criterio que el MLP debe superar. No es evidencia estadística fuerte: no hay incertidumbre por semillas para Ridge y el test temporal elegible es de 2.773 filas.

`src/metricas_lgd.py` calcula MAE, RMSE, sesgo y R² tras recortar `p` a [0,1]. `pct_fuera_de_rango` en el CSV se calcula aparte sobre la predicción cruda. Ridge guarda predicción cruda y recortada en parquet. Los modelos `.joblib` son objetos sklearn `Ridge`; el CSV tiene 12 filas (media y Ridge × tres conjuntos × dos particiones); el parquet tiene 92.018 filas de predicción (castigados repetidos bajo dos protocolos) con `id`, partición, conjunto, observado, predicción cruda y recortada.

## Pérdida esperada y contrato de integración

`src/perdida_esperada.py` calcula por crédito `perdida_esperada = pd_hat * lgd_hat * funded_amnt`, donde PD viene de logística y LGD de Ridge recortado; usa `funded_amnt` al otorgamiento como EAD disponible en decisión. Para pérdidas realizadas en incumplidos usa `LGD_obs * ead`, con ead residual calculado desde el principal recibido. Es decir, el EAD de la pérdida esperada está en origen y el de la pérdida realizada al incumplir; el código describe el primero como conservador porque hubo amortización. El desfase entre definiciones debe hacerse explícito al presentar resultados.

La salida crea `results/perdida_esperada_{particion}.parquet`, una tabla de deciles y un JSON con totales. En prueba aleatoria reporta pérdida esperada 119,49 M, realizada mínima 70,86 M y máxima 92,98 M; temporal, 102,53 M, 38,67 M y 110,24 M. Son cifras de los artefactos actuales, afectadas por el límite de censura/missing señalado arriba y por el preprocesador temporal. `src/perdida_esperada.py` espera un Ridge en `models/lgd_lineal_<particion>.joblib` y no ofrece una interfaz genérica de modelos. La app/comparador no debe cambiarse para consumir el MLP hasta terminar su búsqueda y corridas finales, y coordinar el contrato de carga con C.

Hay además una inconsistencia menor: la ayuda superior de `src/perdida_esperada.py` indica `src/expected_loss.py`, pero el archivo real/comando correcto es `src/perdida_esperada.py`.

## Decisión de salida y pérdida para el MLP

El objetivo es una fracción continua acotada con una masa real grande en LGD=1 (46,36 % de los castigados) y solo 0,09 % en LGD=0. Propongo sigmoid en la salida, de modo que la predicción esté en [0,1], y MSE como pérdida inicial. Con MSE, el predictor óptimo aproxima la media condicional de LGD, que es la cantidad que entra en `PD × LGD × EAD`, y alinea el entrenamiento con RMSE, métrica usada por Ridge. MAE, RMSE, sesgo y R² siguen reportándose con la función común.

La sigmoid no produce exactamente 0 ni 1 en aritmética finita; MSE no estima explícitamente la probabilidad de pertenecer a la masa puntual en 1. Por eso esta demo busca primero una media condicional útil, no una distribución predictiva completa. Una extensión posible es un modelo de dos partes (clasificador de recuperación completa + regresor para severidad <1) o una familia zero/one-inflated apropiada. No usaría Beta ordinaria sin tratar las masas en los extremos. La elección de MSE+sigmoid es una hipótesis de baseline, no una decisión final del informe.

## Demo implementada y límites

El comando ejecuta dos redes (una por partición), usa **todas** las filas castigadas de entrenamiento, todas las de validación, las 99 columnas del preprocesador y solo dos épocas. Selecciona el mejor checkpoint con RMSE de validación; no evalúa test ni elige hiperparámetros. Contrasta el resultado de validación con las métricas Ridge y media ya guardadas. Guarda checkpoints, CSV de comparación, predicciones de validación, resumen de distribución, historial, figura histograma/Q-Q y un JSON con versiones, hashes, conteos y tiempo en `results/lgd_mlp_demo/` y `models/lgd_mlp_demo/`.

La demo tomó **3,50 s de pared** (1,26 s de entrenamiento) y usó semilla 2026. Métricas de validación:

| Partición | Modelo | MAE | RMSE | Sesgo | R² | LGD media predicha / observada |
|---|---|---:|---:|---:|---:|---:|
| Aleatoria | MLP demo, 2 épocas | 0,286575 | 0,297763 | −0,280323 | −8,735702 | 0,649693 / 0,930016 |
| Aleatoria | Ridge | 0,072832 | 0,094746 | −0,000865 | 0,014293 | — |
| Aleatoria | Media train | 0,073470 | 0,095436 | −0,000993 | −0,000108 | — |
| Temporal | MLP demo, 2 épocas | 0,330247 | 0,340829 | −0,328575 | −16,650591 | 0,616576 / 0,945151 |
| Temporal | Ridge | 0,077379 | 0,085859 | −0,028311 | −0,120107 | — |
| Temporal | Media train | 0,077350 | 0,085752 | −0,027788 | −0,117326 | — |

Las dos épocas bajan RMSE entre época 1 y 2, pero la red sigue prediciendo muy por debajo de la media observada. La comparación histograma/Q-Q confirma que aún no reproduce la masa en uno: todas las salidas sigmoid son interiores (ninguna exactamente uno) y la mayor parte de predicciones se concentra cerca de 0,62–0,65. Es útil como verificación de end-to-end, no como comparación de desempeño. La demo tampoco demuestra que entrenar más épocas solucione por sí solo el sesgo ni evalúa generalización final.

Comando desde raíz, con `.venv`:

```powershell
.\.venv\Scripts\python.exe scripts/16_entrenar_mlp_lgd.py
```

```bash
python scripts/16_entrenar_mlp_lgd.py
```

No sobrescribe salidas existentes; para repetir, pasa `--output-dir` y `--models-dir` nuevos. La demo verifica la integración y ofrece una comparación exploratoria de validación; dos épocas y una semilla no son evaluación final. No se modificó el trabajo de C.

## Búsqueda y corridas finales pendientes

El YAML documenta nueve variantes: base; cambio aislado de anchura, profundidad, activación, dropout, weight decay, optimizador (emparejado con decay) y dos tasas. Protocolo planeado: 2 semillas exploratorias × 9 configuraciones × 2 particiones = 36 ajustes, máximo 16 épocas, batch 2.048, paciencia 4, presupuesto CPU 30 min. Elegir por menor RMSE medio de validación dentro de cada partición; no mezclar aleatoria y temporal. Reportar MAE, RMSE, sesgo, R², distribución/residuos y tiempo/épocas/parámetros. Test permanece intacto hasta fijar configuraciones.

Después de fijar una configuración por partición, ejecutar 30 semillas (0–29) para ambas particiones, 60 ajustes, presupuesto CPU 45 min. Resumir media, desviación, mejor y peor de MAE/RMSE/sesgo/R² en validación y prueba; no reajustar según test. El CPU observado es Windows con PyTorch `2.8.0+cpu`, dos hilos. La arquitectura base tiene **3.233** parámetros (99→32→1 más biases); la variante 64 tiene 6.465; [32,16] tiene 3.745. La demo mide el costo de dos épocas completas y permitirá ajustar presupuestos antes de implementar la búsqueda/final. Actualmente búsqueda y 30 semillas no están implementadas ni ejecutadas.

## Estado de requisitos

| Componente | Estado | Evidencia / límite |
|---|---|---|
| Objetivo/derivación LGD y EAD | Parcial | Código y diccionario concordantes; CSV fuente ausente impide recomputar registro por registro. |
| Muestra de Ridge | Hecho | 46.009 castigados; IDs de train/validación/prueba coinciden con predicciones Ridge. |
| Exclusión de no castigados censurados | Hecho | 12.805, todos LGD=1, documentados como censurados/no maduros por estado. |
| Ridge y media de train | Hecho | Modelos, CSV, predicciones y métricas leídos; temporalmente media gana ligeramente. |
| Lista exacta de alphas Ridge | Pendiente | Resumen dice ocho; fuente da seis por defecto; alpha temporal 100.000. |
| MLP configurable y demo | Hecho tras ejecución | Código/config nuevos, validation-only, artefactos nuevos; resultados demo no finales. |
| Búsqueda MLP / 30 semillas | Pendiente | Diseño documentado, todavía no ejecutado. |
| Uso del MLP en expected-loss/app | Pendiente | Consumidor actual está acoplado a `.joblib` Ridge; coordinar contrato con C. |
| Fuga temporal del preprocesador | Parcial/limitación abierta | Base registrada `entrenamiento_aleatorio`; no prueba ausencia de fuga temporal. |
| Cota superior de pérdida realizada | Parcial | 5 filas PD=1 sin LGD; en test son 1 aleatoria y 2 temporal y se contabilizan como censurados con pérdida cero. |
