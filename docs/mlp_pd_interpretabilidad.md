# Interpretabilidad de la MLP de PD

## Alcance y método

Se analizaron los modelos finales ya entrenados de PD, sin volver a ajustarlos. Para
cada partición (aleatoria y temporal) se promediaron las predicciones de sus 30
checkpoints, semillas 0–29. El análisis usa únicamente las filas de validación de
esa partición: 40.280 aleatorias y 74.154 temporales. La prueba no intervino en el
cálculo de importancias ni en las tablas descriptivas.

El preprocesador convierte 29 columnas de entrada en 99 características. Estas se
reunieron en 24 grupos semánticos: las columnas one-hot de una categoría se permutan
juntas y cada indicador de ausencia se permuta con su variable numérica. Para cada
grupo se permutaron cinco veces sus filas con semilla fija 20261027. La importancia
principal es el aumento de log-loss frente a la predicción sin permutar. Brier y
AUC son diagnósticos adicionales. El método cuantifica la dependencia predictiva
en esta muestra; no demuestra causalidad ni permite inferir el signo del efecto.
Los predictores correlacionados pueden repartirse la importancia o generar
combinaciones poco plausibles al permutarlos. Véase la
[documentación de scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html).

## Resultados principales

| Partición | Variable | Aumento de log-loss, media ± DE |
|---|---|---:|
| Aleatoria | `annual_inc` | 0,01815 ± 0,00089 |
| Aleatoria | `term_meses` | 0,01771 ± 0,00066 |
| Aleatoria | `loan_amnt` | 0,01537 ± 0,00079 |
| Aleatoria | `revol_util` | 0,01204 ± 0,00036 |
| Aleatoria | `open_acc` | 0,00814 ± 0,00043 |
| Temporal | `term_meses` | 0,02080 ± 0,00048 |
| Temporal | `annual_inc` | 0,02006 ± 0,00028 |
| Temporal | `loan_amnt` | 0,01087 ± 0,00054 |
| Temporal | `revol_util` | 0,00930 ± 0,00016 |
| Temporal | `open_acc` | 0,00899 ± 0,00058 |

Los cinco grupos principales coinciden entre particiones, con distinto orden y
magnitud. La figura de los 24 grupos está en
`results/pd_mlp_interpretabilidad_v2/importancia_agrupada_validacion.png`.
La desviación estándar de esta tabla corresponde a las cinco permutaciones del
**ensamble**, no a la variación entre 30 modelos. Como diagnóstico separado se
evaluaron las semillas 0, 7, 14, 21 y 29, fijadas por espaciamiento y no por
desempeño; el detalle está en `estabilidad_por_semilla.csv`.

Para H1–H8 se compararon grupos de créditos observados dentro de validación. En
ambas particiones, las diferencias de PD media predicha, las tasas observadas y
los signos de los coeficientes logísticos apuntan en la dirección planteada. Son
asociaciones marginales: los grupos también pueden diferir en otras variables.
No son efectos individuales controlados ni pruebas causales. Los soportes y
promedios de cada grupo están en `hipotesis_por_grupos_validacion.csv`; la tabla
integrada con la logística está en `tabla_hipotesis/tabla_hipotesis_final.csv`.
La columna `coincide_logistica` se refiere solo al signo de la logística.

## Verificación y reproducción

La repetición del 27 de octubre de 2026 se guardó en una carpeta nueva (`v2`).
Sus cuatro CSV principales, dos CSV de estabilidad, dos CSV de hipótesis y el CSV
de verificación por semilla son **numéricamente idénticos** a la primera ejecución
(`v1`, conservada localmente). Las métricas de validación recalculadas para los 60
checkpoints coinciden con las publicadas: error absoluto máximo 1,99 × 10⁻¹²,
frente a una tolerancia de 10⁻⁶. El manifiesto de `v2` registra los hashes de los
datos, el preprocesador, los modelos, el script de interpretación y `src/mlp_pd.py`.
La ejecución principal tardó 109,6 segundos en esta repetición.

Desde la raíz del repositorio, con `.venv` instalada, usar una carpeta nueva al
reproducir; los scripts no sobrescriben salidas existentes:

```powershell
.\.venv\Scripts\python.exe scripts/24_interpretabilidad_mlp_pd.py --output-dir results/pd_mlp_interpretabilidad_reproduccion --repeats 5 --num-threads 2
.\.venv\Scripts\python.exe scripts/25_estabilidad_importancia_mlp_pd.py --output-dir results/pd_mlp_interpretabilidad_reproduccion --num-threads 2
.\.venv\Scripts\python.exe scripts/27_verificar_metricas_validacion_pd_mlp.py --output-dir results/pd_mlp_interpretabilidad_reproduccion --atol 1e-6
.\.venv\Scripts\python.exe scripts/12_tabla_hipotesis_final.py --importance-file results/pd_mlp_interpretabilidad_reproduccion/importancia_agrupada_validacion.csv --output-dir results/pd_mlp_interpretabilidad_reproduccion/tabla_hipotesis
```

La evaluación temporal conserva la limitación conocida: el preprocesador común
se ajustó sobre entrenamiento aleatorio, que puede solaparse con observaciones
temporales posteriores. Los resultados temporales deben leerse bajo esa salvedad.
Las métricas del ensamble promedio no deben confundirse con el promedio de las
métricas individuales de los 30 checkpoints.
