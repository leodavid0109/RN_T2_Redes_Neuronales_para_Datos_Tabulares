# Registro del entorno y balance de PD

Fecha de verificación: 25 de octubre de 2026. Las verificaciones de datos y modelos son de solo lectura; no se regeneraron modelos, tablas ni figuras.

## Entorno

### Inspección previa

- `git status --short --branch` estaba limpio en `main` antes de los cambios de esta tarea.
- El `.gitignore` ya ignoraba `.venv/`, pero la regla se hizo explícita y anclada a la raíz como `/.venv/` para no ocultar otros directorios del proyecto.
- `py -0p` solo mostró Python 3.14 instalado para el lanzador de Windows; el comando `python` apuntaba al alias de Microsoft Store y no iniciaba Python.
- El runtime local de Codex proporcionó Python 3.12.14 en `C:\Users\david\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`. Ese intérprete creó `.venv`; su `pyvenv.cfg` confirma Python 3.12.14.
- Los JSON existentes registran que los resultados/modelos se generaron con Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, pyarrow 25.0.1 y joblib 1.6.0. Se mantuvieron esas versiones de ciencia de datos.
- Inspeccioné imports en `src/`, `scripts/` y `tests/`. Añadí matplotlib para las figuras, PyYAML para la configuración de scorecard, openpyxl para leer el diccionario y PyTorch para los MLP. No añadí dependencias de app porque no hay código de aplicación. La prueba existente se ejecuta directamente, por eso no se añadió pytest.

### Instalación realizada

La lista completa de versiones directas y transitivas está en `requirements.txt`. PyTorch se fijó a `2.8.0+cpu`, desde el índice CPU oficial, para que el entorno funcione igual en Windows y Linux sin CUDA ni ROCm. El equipo auditado muestra un AMD Ryzen 7 5700G; no se identificó una GPU NVIDIA/CUDA configurada.

Comandos ejecutados en Windows:

```powershell
& 'C:\Users\david\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
$tmpPath = Join-Path (Get-Location) '.task-tmp-python'
New-Item -ItemType Directory -Path $tmpPath -Force | Out-Null
$env:TEMP = $tmpPath
$env:TMP = $tmpPath
& .\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip
& .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
```

`ensurepip` necesitó redirigir `TEMP` y `TMP` a una carpeta temporal dentro del workspace. La primera instalación en modo sandbox no pudo resolver el índice de paquetes; se completó el comando autorizado con acceso de red. El resultado final fue `Successfully installed` con todas las versiones fijadas y `pip check` respondió `No broken requirements found`.

La versión instalada en `.venv` fue:

| Paquete | Versión |
|---|---:|
| Python | 3.12.14 |
| pip | 25.0.1 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| pyarrow | 25.0.1 |
| joblib | 1.6.0 |
| matplotlib | 3.10.7 |
| PyYAML | 6.0.3 |
| openpyxl | 3.1.5 |
| PyTorch | 2.8.0+cpu |

`torch.cuda.is_available()` devolvió `False`; una multiplicación matricial de CPU devolvió 5.0. Importé todas las dependencias fijadas. La primera importación de matplotlib avisó que el directorio de configuración predeterminado del usuario no es escribible en este sandbox; matplotlib pudo importar y ese aviso no bloqueó la instalación ni las pruebas.

### Comprobaciones seguras

| Comprobación | Comando/entrada | Resultado |
|---|---|---|
| Scorecard | `.venv/Scripts/python.exe tests/test_scorecard.py` | Pasaron las 5 pruebas: PDO, odds de referencia, factor/offset, ida y vuelta/monotonía, contribuciones y escenarios. |
| Parquet | Carga de `data/processed/pd.parquet` con pandas/pyarrow | Leído; forma `(268530, 34)`. Etiquetas: 209.711 clase 0 y 58.819 clase 1. |
| Preprocesador | `joblib.load(models/preprocessor.joblib)` y `transform` de 16 filas | Cargó como `ColumnTransformer`; 29 columnas de entrada, 99 transformadas. |
| Logística aleatoria | `joblib.load(models/pd_logreg_aleatoria.joblib)` y `predict_proba` | Cargó como `LogisticRegression`, coeficientes `(1, 99)`; produjo probabilidades para la muestra. |
| Logística temporal | `joblib.load(models/pd_logreg_temporal.joblib)` y `predict_proba` | Cargó como `LogisticRegression`, coeficientes `(1, 99)`; produjo probabilidades para la muestra. |
| Consistencia de paquetes | `.venv/Scripts/python.exe -m pip check` | `No broken requirements found`. |
| Hardware portátil | Extraje y ejecuté solo `_nombre_cpu` del AST de `scripts/08_regresion_logistica.py` | No intentó leer `/proc/cpuinfo` en Windows; devolvió `AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD`. |

No ejecuté `scripts/06_generar_datos_pd.py`, `scripts/08_regresion_logistica.py`, `scripts/09_guardar_logistica.py` ni `scripts/10_scorecard.py`, pues escribirían o regenerarían resultados/modelos. La prueba de scorecard y el script de balance son de solo lectura.

### Compatibilidad al recrear

Este host no tiene Python 3.12 registrado en `py` aunque el runtime de Codex sí lo proporciona; por eso el comando de creación local usado arriba tiene una ruta propia de este equipo. En otra máquina se requiere instalar Python 3.12.14 y usar `py -3.12 -m venv .venv` en Windows o `python3.12 -m venv .venv` en Linux. No se debe copiar `.venv` entre sistemas. La recreación de paquetes se hace con `.venv` activado y `python -m pip install -r requirements.txt`.

La instalación y comprobaciones se ejecutaron en Windows. Linux está documentado pero no fue probado en este host; el archivo fija la misma CPU build de PyTorch para ambos sistemas.

## Balance de clases PD

Reproducción, sin escritura de archivos:

```powershell
.\.venv\Scripts\python.exe scripts/13_balance_clases_pd.py
```

| Partición | Conjunto | Registros | Clase 0 | % clase 0 | Clase 1 | % clase 1 |
|---|---|---:|---:|---:|---:|---:|
| Aleatoria estratificada | Entrenamiento | 187.970 | 146.797 | 78,10 | 41.173 | 21,90 |
| Aleatoria estratificada | Validación | 40.280 | 31.457 | 78,10 | 8.823 | 21,90 |
| Aleatoria estratificada | Prueba | 40.280 | 31.457 | 78,10 | 8.823 | 21,90 |
| Temporal | Entrenamiento (hasta 2013) | 163.541 | 132.014 | 80,72 | 31.527 | 19,28 |
| Temporal | Validación (2014) | 74.154 | 54.713 | 73,78 | 19.441 | 26,22 |
| Temporal | Prueba (2015) | 30.835 | 22.984 | 74,54 | 7.851 | 25,46 |

La clase 1 es minoritaria en todas las particiones. En la aleatoria hay cerca de 3,57 créditos de clase 0 por cada clase 1; temporalmente la relación cambia de aproximadamente 4,19:1 en entrenamiento a 2,81:1 en validación y 2,93:1 en prueba. La etiqueta positiva no es extremadamente rara (19–26 %), pero la diferencia justifica probar como experimento una alternativa ponderada o remuestreada y medir el efecto.

### Estratificación no es balanceo

La partición aleatoria usa `stratify=y` en `scripts/06_generar_datos_pd.py`: procura que cada split conserve la proporción de clases del total. No cambia el número de ejemplos, no duplica la clase 1 y no descarta la clase 0. La partición temporal no estratifica: deja que la prevalencia de cada período aparezca como es para medir el cambio temporal.

Balancear durante el ajuste sería distinto. `class_weight=None` en la logística (`scripts/08_regresion_logistica.py`), y el código disponible no implementa sobremuestreo ni submuestreo. No hay aún MLP en el repositorio. Por tanto, el entrenamiento actual no aplica balanceo, aunque la división aleatoria sí sea estratificada.

### Decisión experimental

No se activa balanceo automáticamente. El modelo actual muestra una base útil de comparación: aleatoria, prueba AUC 0,6918 / KS 0,2823 / Brier 0,1572; temporal, prueba AUC 0,6749 / KS 0,2582 / Brier 0,1773. El salto de prevalencia temporal (19,28 % en entrenamiento a 26,22 % en validación) es un motivo para investigar cómo cambian las probabilidades, pero no demuestra que ponderar mejore el modelo.

Si se prueba balanceo:

1. Mantener los mismos registros, particiones y preprocesamiento del control sin balanceo.
2. Calcular pesos de clase o hacer remuestreo solo dentro de cada fold/conjunto de entrenamiento. No balancear ni ponderar validación o prueba.
3. Comparar control y tratamiento con el mismo presupuesto de búsqueda y semillas; mantener validación y prueba en sus proporciones naturales.
4. Reportar AUC y KS para ordenamiento, y Brier más curvas de calibración para la calidad de las probabilidades. La ponderación/remuestreo puede mejorar el énfasis en la clase minoritaria, pero desplazar probabilidades respecto a la prevalencia natural; por eso AUC/KS por sí solos no deciden.
5. Usar validación para seleccionar y evaluar calibración; reservar la prueba para una evaluación final. No ajustar corrección de prevalencia o calibración con prueba.

No se recomienda submuestrear de entrada: descartaría datos mayoritarios disponibles. Una comparación ponderada, sin cambiar filas, puede ser más informativa como primer experimento.

### Responsables y coordinación

- **A (datos/Partición 1):** define etiqueta, variables disponibles al otorgar, particiones y corrección del ajuste del preprocesador. No cambiar split o prevalencias naturales de validación/prueba para facilitar al MLP.
- **B (protocolo/redes):** decide si compara el MLP con y sin ponderación/remuestreo; contabiliza esa configuración en búsqueda/tiempos/semillas y evalúa con métricas de discriminación y calibración.
- **Cambio compartido:** si se quiere afirmar que el efecto de balanceo es comparable entre familias, coordinar con quien implementa la logística una referencia ponderada correspondiente. A y B deben usar exactamente las mismas particiones, transformación y definición de pesos; las predicciones de ambos tratamientos se evalúan sobre datos naturales. C integra el contrato de configuración/resultados y el reporte.

## MLP de probabilidad de incumplimiento (Persona B)

### Qué implementé y por qué

`src/mlp_pd.py` define una red binaria que recibe las 99 columnas transformadas y emite un único **logit** por fila. Un logit es un puntaje sin acotar; `sigmoid(logit)` lo convierte en un número de 0 a 1 que el código interpreta explícitamente como `P(objetivo=1)`, es decir, probabilidad de incumplimiento. Durante el ajuste se usa `BCEWithLogitsLoss`, que calcula la pérdida Bernoulli de forma numéricamente estable sin aplicar sigmoid por separado.

`scripts/14_entrenar_mlp_pd.py` carga el mismo parquet y `models/preprocessor.joblib` que usa `scripts/08_regresion_logistica.py`. Lee las columnas desde `pre.feature_names_in_`, rechaza objetivo o máscaras de split si se colaran como predictores, transforma una sola vez y luego corta esas mismas filas por `split_random` y `split_temporal`. La logística transforma a su vez el mismo parquet con el mismo artefacto y las mismas columnas; el MLP convierte su matriz resultante a `float32` para PyTorch, por lo que la representación y filas son compartidas, con la diferencia normal de precisión de almacenamiento frente a la matriz de scikit-learn.

Esta decisión evita duplicar reglas de imputación y codificación en la red: ambos modelos comparan las mismas entradas. No corrige el ajuste temporal defectuoso del preprocesador actual; de hecho, la verificación existente confirma que se ajustó con `entrenamiento_aleatorio`, que incluye observaciones de validación y prueba temporales. Por eso todos los valores de esta sección son **provisionales**.

La configuración vive en `configs/mlp_pd.yaml`, no en constantes del código. La búsqueda es deliberadamente pequeña y cambia:

| ID | Capas | Activación | Regularización | Optimizador | Tasa |
|---|---|---|---|---|---:|
| `shallow32_relu_adam` | 32 | ReLU | dropout 0; weight decay 1e-4 | Adam | 0,001 |
| `deep64_32_relu_dropout_adam` | 64, 32 | ReLU | dropout 0,2; sin weight decay | Adam | 0,001 |
| `shallow64_tanh_l2_adamw` | 64 | tanh | weight decay 1e-3 | AdamW | 0,0005 |
| `deep32_16_gelu_l2_sgd` | 32, 16 | GELU | dropout 0,1; weight decay 1e-4 | SGD | 0,01 |

Dropout apaga al azar algunas activaciones durante el entrenamiento para limitar dependencia excesiva entre neuronas. `weight_decay` penaliza pesos grandes (regularización L2 en Adam; AdamW aplica su variante desacoplada). La búsqueda elige **una configuración global** minimizando el promedio simple de log-loss de validación aleatoria y temporal. La log-loss premia probabilidades que asignan alta probabilidad a la etiqueta observada y penaliza con fuerza errores confiados; por eso ayuda a escoger un modelo probabilístico. AUC y KS miden ordenamiento, mientras que Brier y la curva de calibración ayudan a detectar probabilidades mal calibradas. La prueba nunca participa en selección; solo se evalúa una vez fijada la configuración.

El ajuste es sin balanceo (`class_weight: none`). El DataLoader baraja solo los datos de entrenamiento con semilla controlada; la red, NumPy y PyTorch reciben la misma semilla por corrida. La parada temprana compara log-loss no ponderado en validación, guarda en memoria el estado con menor valor y lo restaura antes de evaluar. El checkpoint final almacena configuración, semilla, dimensión, columnas, estado de pesos y la convención de clase. También se reportan cantidad de parámetros y segundos de entrenamiento por corrida. El dispositivo de esta entrega es CPU (`torch 2.8.0+cpu`), fijado para reproducibilidad entre Windows/Linux y porque no hay una GPU CUDA configurada en el equipo inspeccionado.

### Riesgo del protocolo de selección global

La búsqueda implementada calcula un promedio de log-loss entre validación aleatoria y validación temporal para elegir **una arquitectura global**. Las máscaras de ambas particiones no son independientes ni disjuntas. Verifiqué en el parquet que **4.540** filas de validación aleatoria pertenecen a prueba temporal, y **11.154** filas de prueba aleatoria pertenecen a validación temporal. Por tanto, una selección que consulta ambas validaciones usa filas que son prueba en el otro protocolo. La arquitectura escogida por esa búsqueda podría quedar expuesta a datos de prueba y su evaluación posterior no sería una comparación final limpia.

`--mode search` conserva esta selección global solo como resultado exploratorio/provisional. El resumen lo marca `seleccion_final_habilitada: false`, y `--mode final` rechaza usar ese manifiesto. No cambies esa protección ni uses la arquitectura resultante como definitiva hasta que el equipo defina y documente una selección que respete la independencia de la prueba para ambos protocolos. Una alternativa a evaluar por el equipo es escoger y reportar configuraciones por protocolo, o establecer datos de selección comunes que no pertenezcan a ninguna prueba; la decisión debe coordinarse con A/C y aplicarse de manera comparable a la referencia logística.

### Corrida de extremo a extremo verificada

Comando ejecutado desde la raíz con el Python del entorno:

```powershell
Measure-Command { .\.venv\Scripts\python.exe scripts/14_entrenar_mlp_pd.py --mode demo }
```

La ejecución usó **todas** las filas disponibles de cada split, una arquitectura pequeña 99→32→1 y un máximo de dos épocas. No fue necesario muestrear registros. PowerShell midió 14,718 segundos de pared; el resumen del script registró 9,249 segundos incluyendo carga, entrenamiento y evaluación, de los cuales 7,135 segundos fueron las dos corridas de entrenamiento. La demo verifica ejecución e integración, no desempeño final. El resultado quedó únicamente en `results/pd_mlp_demo/` y `models/pd_mlp_demo/`:

| Partición | Conjunto | AUC | KS | Brier | Log-loss |
|---|---|---:|---:|---:|---:|
| Aleatoria | Validación | 0,6859 | 0,2783 | 0,1585 | 0,4885 |
| Aleatoria | Prueba | 0,6865 | 0,2759 | 0,1580 | 0,4880 |
| Temporal | Validación | 0,6756 | 0,2576 | 0,1810 | 0,5427 |
| Temporal | Prueba | 0,6738 | 0,2612 | 0,1776 | 0,5361 |

La salida guardada reporta `sigmoid(logit) = P(objetivo=1 = incumplimiento)`; las probabilidades se verificaron dentro de [0,1]. Verifiqué cuatro filas métricas (semilla 919 × dos particiones × validación/prueba), 40 grupos de calibración (10 por combinación), columnas compatibles con el contrato de `scripts/11_tabla_comparacion_pd.py`, y dos checkpoints que pudieron abrirse con `torch.load(..., weights_only=True)`. Cada red tiene 3.233 parámetros entrenables. Las métricas solo muestran que el cableado produce valores y archivos coherentes; dos épocas y una semilla no son evaluación del enunciado. No corrí el comparador 11 con esta demo porque requiere al menos 30 semillas, ni entrené la búsqueda completa ni las 30 semillas definitivas.

Archivos producidos por la demo:

- `results/pd_mlp_demo/pd_mlp_metricas_por_semilla.csv`
- `results/pd_mlp_demo/curva_calibracion_mlp.csv`
- `results/pd_mlp_demo/pd_mlp_busqueda.csv`
- `results/pd_mlp_demo/pd_mlp_resumen.json` (incluye hashes de los artefactos de entrada y versiones)
- `models/pd_mlp_demo/pd_mlp_aleatoria_demo_seed_919.pt`
- `models/pd_mlp_demo/pd_mlp_temporal_demo_seed_919.pt`

El script se niega a escribir sobre carpetas de salida que ya contengan archivos y no sobrescribe un checkpoint existente. La demo ya ocupa sus carpetas por defecto; para repetirla usa salidas nuevas, por ejemplo `--output-dir results/pd_mlp_demo_2 --models-dir models/pd_mlp_demo_2`.

### Comandos y protocolo posterior

Windows PowerShell, después de activar `.venv`:

```powershell
python scripts/14_entrenar_mlp_pd.py --mode demo
python scripts/14_entrenar_mlp_pd.py --mode search
```

Linux, después de activar `.venv`:

```bash
python scripts/14_entrenar_mlp_pd.py --mode demo
python scripts/14_entrenar_mlp_pd.py --mode search
```

La búsqueda provisional entrena las cuatro configuraciones en entrenamiento y validación de las dos particiones; al final reevalúa validación y prueba solo para la configuración seleccionada. Su contrato de resultados conserva los nombres de columnas esperados por el script 11. **No se ha ejecutado la búsqueda válida ni las 30 semillas finales.** El código exige que la verificación de datos diga `preprocesador_ajustado_con: interseccion`, que se haya generado primero la búsqueda con los mismos hashes de datos, preprocesador, logística y configuración, el indicador explícito `--confirmar-frente-datos` y `seleccion_final_habilitada: true`. El protocolo actual siempre lo marca `false` debido al solapamiento; la protección requiere corregir la estrategia de selección además de regenerar datos. A debe confirmar/regenerar datos y particiones, preprocesador ajustado exclusivamente con entrenamiento común y modelos/resultados logísticos correspondientes. Los resultados temporales deben repetirse cuando A corrija el ajuste del preprocesador y regenere la logística. Luego debe acordarse una búsqueda sin exposición de prueba entre particiones y regenerarse esa búsqueda antes de lanzar 30 semillas.

La demo de 14,7 s verifica una ruta corta, no garantiza que la búsqueda completa termine en menos de dos minutos: esta última prueba 4 configuraciones en ambos splits y usa hasta 40 épocas con parada temprana. La limitación adicional del preprocesador actual invalida cualquier interpretación de una comparación final, aunque el MLP y la logística reciban exactamente la misma transformación guardada.
