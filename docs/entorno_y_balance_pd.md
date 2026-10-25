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

### Decisión comunicada sobre el preprocesador de A

Busqué en código, documentación, metadatos JSON y artefactos del repositorio un informe de la validación de A. No encontré el nombre exacto de la alternativa evaluada, sus datos, métricas, tabla de diferencias ni criterio de aceptación. La evidencia guardada sigue registrando `preprocesador_ajustado_con: entrenamiento_aleatorio`; `scripts/06_generar_datos_pd.py` conserva una opción `interseccion`, pero eso no prueba que se haya probado ni cuál fue el resultado.

Por información comunicada por el equipo en esta conversación, A evaluó una alternativa y el equipo decidió conservar por ahora `data/processed/pd.parquet` y `models/preprocessor.joblib` actuales porque la mejora observada no justificó incorporarla. No registro cifras, definición exacta de la alternativa ni resultado cuantitativo porque no están en el repositorio ni fueron especificados aquí. La aceptación es una decisión de trabajo; **no demuestra ausencia de fuga temporal**.

La verificación existente dice que el preprocesador se ajustó con entrenamiento aleatorio, que contiene registros que luego forman validación y prueba temporal (2014/2015). Los resultados temporales de estos experimentos conservan esta limitación. El acuerdo de equipo permite continuar sin esperar una regeneración de A.

### Qué implementé y por qué

`src/mlp_pd.py` define una red binaria que recibe las 99 columnas transformadas y emite un único **logit** por fila. Un logit es un puntaje sin acotar; `sigmoid(logit)` lo convierte en un número de 0 a 1 que el código interpreta explícitamente como `P(objetivo=1)`, es decir, probabilidad de incumplimiento. Durante el ajuste se usa `BCEWithLogitsLoss`, que calcula la pérdida Bernoulli de forma numéricamente estable sin aplicar sigmoid por separado.

`scripts/14_entrenar_mlp_pd.py` carga el mismo parquet y `models/preprocessor.joblib` que usa `scripts/08_regresion_logistica.py`. Lee las columnas desde `pre.feature_names_in_`, rechaza objetivo o máscaras de split si se colaran como predictores, transforma una sola vez y luego corta esas mismas filas por `split_random` y `split_temporal`. La logística transforma a su vez el mismo parquet con el mismo artefacto y las mismas columnas; el MLP convierte su matriz resultante a `float32` para PyTorch, por lo que la representación y filas son compartidas, con la diferencia normal de precisión de almacenamiento frente a la matriz de scikit-learn.

Esta decisión evita duplicar reglas de imputación y codificación en la red: ambos modelos comparan las mismas entradas. No corrige el ajuste temporal defectuoso del preprocesador actual; de hecho, la verificación existente confirma que se ajustó con `entrenamiento_aleatorio`, que incluye observaciones de validación y prueba temporales. Por eso todos los valores de esta sección son **provisionales**.

La configuración vive en `configs/mlp_pd.yaml`, no en constantes del código. La tabla de hipótesis y las variantes se fijaron antes de ejecutar búsqueda:

| ID | Configuración frente a base | Hipótesis y motivo |
|---|---|---|
| `base_32_relu_adam` | 99→32→1; ReLU; dropout/WD 0; Adam; lr 0,001 | Referencia pequeña para comparar. |
| `width64` | Anchura 32→64 | Más unidades podrían representar interacciones no lineales. |
| `depth32_16` | Una capa pasa a dos: [32,16] | Prueba composición jerárquica; también cambia parámetros y unidades totales. |
| `activation_tanh` | ReLU→tanh | Contrasta una no linealidad saturante. |
| `dropout_020` | Dropout 0→0,20 | Prueba regularización estocástica contra sobreajuste. |
| `weight_decay_1e4` | WD 0→1e-4 | Prueba penalización de pesos grandes. |
| `optimizer_adamw` | AdamW y WD 1e-4; comparar con Adam/WD 1e-4 | Aísla AdamW contra su control emparejado de Adam. |
| `optimizer_sgd` | Adam→SGD con momentum 0,9; misma tasa | Referencia no adaptativa; no es búsqueda exhaustiva de SGD. |
| `learning_rate_0003` | lr 0,001→0,0003 | Actualización más conservadora. |
| `learning_rate_003` | lr 0,001→0,003 | Actualización más rápida, con posible inestabilidad. |
| `combo_width64_dropout020` | Anchura 64 + dropout 0,20 | Interacción de capacidad/regularización; no atribuir a un solo factor. |
| `combo_depth_weight_decay` | [32,16] + WD 1e-4 | Interacción de profundidad/penalización; no atribuir a un solo factor. |

El presupuesto declarado es 12 configuraciones × 2 semillas exploratorias (2026, 2027) × 2 particiones = 48 ajustes máximos; hasta 16 épocas, batch 2.048 y paciencia 4. El primer límite propuesto fue 30/40 minutos CPU para búsqueda/final. Un piloto interrumpido tras 6 ajustes aleatorios consumió 385 s de CPU, así que revisé el presupuesto **antes de la búsqueda completa** a 60 minutos CPU para búsqueda y 75 para 30 semillas × dos particiones. Los seis checkpoints de ese piloto permanecen en `models/pd_mlp_search_particionado/`; no se guardó tabla de búsqueda ni se eligió arquitectura con esos ajustes. La validación se consultó únicamente para early stopping, como requiere el protocolo. La elección predefinida es la menor media de log-loss de las dos semillas dentro de **cada partición**. Log-loss es el criterio principal porque se entregan probabilidades; AUC/Gini/KS describen ordenamiento y Brier/calibración describen calidad probabilística. Las pruebas no intervienen en búsqueda.

Dropout apaga activaciones al azar durante el entrenamiento. Weight decay penaliza pesos grandes; Adam incorpora ese término al gradiente y AdamW desacopla la actualización de decay de la adaptación del gradiente. La selección actual elige independientemente una configuración por partición; sus valores pueden coincidir. La validación aleatoria nunca se promedia con la temporal, y viceversa. La log-loss evalúa cuánto castigan las probabilidades los errores confiados. AUC/Gini/KS describen ordenamiento; Brier y curvas por decil describen calibración.

### Texto técnico adaptable al informe

Las 29 columnas originales definidas por `models/preprocessor.joblib` se transforman una sola vez a 99 características numéricas/categóricas; la logística consume esas mismas filas y columnas. El MLP base es una capa oculta de 32 unidades ReLU y salida de una unidad: `x → Linear(99,32) → ReLU → Linear(32,1) → z`. Variantes de capas, activación y regularización son parámetros del YAML.

La salida `z` es un logit y no una probabilidad. `p = σ(z) = 1/(1+exp(-z))` se interpreta como PD. Bajo una variable Bernoulli, `P(y|p)=p^y(1-p)^(1-y)`. Su log-verosimilitud negativa media es `-1/N Σ[y log σ(z)+(1-y) log(1-σ(z))]`, que es `BCEWithLogitsLoss`; esta forma calcula la combinación de sigmoide y entropía de modo estable. Sin ponderación, cada fila contribuye por igual.

El entrenamiento divide los ejemplos de train en lotes de 2.048. Para cada lote calcula la pérdida, retropropaga su gradiente por la regla de la cadena desde salida hasta entrada y actualiza los pesos. Adam mantiene promedios exponenciales `m_t=β₁m_{t-1}+(1-β₁)g_t`, `v_t=β₂v_{t-1}+(1-β₂)g_t²`, corrige sus sesgos iniciales y actualiza `θ_t=θ_{t-1}-α m̂_t/(√v̂_t+ε)`; con decay desacoplado AdamW añade `-αλθ` a la actualización. SGD con momentum usa `u_t=0,9u_{t-1}+g_t` y `θ_t=θ_{t-1}-αu_t` (el código de PyTorch incorpora weight decay al gradiente si se activa). Dropout se aplica durante train, no durante inferencia; weight decay penaliza pesos según el optimizador configurado.

Después de cada época calcula log-loss en la validación de **esa** partición. Si no mejora al menos `min_delta` durante `patience` épocas, detiene el ajuste y restaura el estado con menor log-loss. La búsqueda de 12 configuraciones usa dos semillas exploratorias y el promedio de log-loss únicamente dentro de cada split. El procedimiento final vuelve a ajustar la configuración ya fijada con 30 semillas por split y evalúa la prueba; no se reajusta tras ver esas métricas. Se reportan también AUC, Gini, KS, Brier y curvas de calibración para mostrar ordenamiento y calidad de probabilidad.

El ajuste es sin balanceo (`class_weight: none`). El DataLoader baraja solo los datos de entrenamiento con semilla controlada; la red, NumPy y PyTorch reciben la misma semilla por corrida. La parada temprana compara log-loss no ponderado en validación, guarda en memoria el estado con menor valor y lo restaura antes de evaluar. El checkpoint final almacena configuración, semilla, dimensión, columnas, estado de pesos y la convención de clase. También se reportan cantidad de parámetros y segundos de entrenamiento por corrida. El dispositivo de esta entrega es CPU (`torch 2.8.0+cpu`), fijado para reproducibilidad entre Windows/Linux y porque no hay una GPU CUDA configurada en el equipo inspeccionado.

### Cruce entre protocolos y corrección de selección

La auditoría del parquet confirmó **4.540** filas de validación aleatoria dentro de prueba temporal y **11.154** filas de prueba aleatoria dentro de validación temporal. Por eso la búsqueda anterior, que promediaba ambas validaciones, exponía filas de prueba de un protocolo al escoger hiperparámetros. La demo previa además calculó métricas de prueba por razones técnicas; quedan registradas como exposición, sin usarlas para elegir arquitectura ni hiperparámetros.

La nueva búsqueda elige independientemente por partición. Para random se compara cada candidato exclusivamente sobre `split_random=validacion`; para temporal, exclusivamente sobre `split_temporal=validacion`. La función no recibe métricas de prueba y el código verifica que train/validation/test sean disjuntos dentro de cada partición. El resumen guarda cada configuración, criterio y conteos de cruce. Ya no se requiere que A regenere el preprocesador para iniciar búsqueda o final; los hashes conservan el contrato exacto de artefactos.

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

La salida guardada reporta `sigmoid(logit) = P(objetivo=1 = incumplimiento)`; las probabilidades se verificaron dentro de [0,1]. Verifiqué cuatro filas métricas (semilla 919 × dos particiones × validación/prueba), 40 grupos de calibración (10 por combinación), columnas compatibles con el contrato de `scripts/11_tabla_comparacion_pd.py`, y dos checkpoints que pudieron abrirse con `torch.load(..., weights_only=True)`. Cada red tiene 3.233 parámetros entrenables. Las métricas solo muestran que el cableado produce valores y archivos coherentes; dos épocas y una semilla no son evaluación del enunciado. **En esa corrida demo** no ejecuté el comparador 11 ni la búsqueda completa/30 semillas; estas dos últimas se completaron después y están reportadas abajo. Las cifras de prueba de la demo quedan solo como evidencia de exposición técnica, no entraron en la selección.

Archivos producidos por la demo:

- `results/pd_mlp_demo/pd_mlp_metricas_por_semilla.csv`
- `results/pd_mlp_demo/curva_calibracion_mlp.csv`
- `results/pd_mlp_demo/pd_mlp_busqueda.csv`
- `results/pd_mlp_demo/pd_mlp_resumen.json` (incluye hashes de los artefactos de entrada y versiones)
- `models/pd_mlp_demo/pd_mlp_aleatoria_demo_seed_919.pt`
- `models/pd_mlp_demo/pd_mlp_temporal_demo_seed_919.pt`

El script se niega a escribir sobre carpetas de salida que ya contengan archivos y no sobrescribe un checkpoint existente. La demo ya ocupa sus carpetas por defecto; para repetirla usa salidas nuevas, por ejemplo `--output-dir results/pd_mlp_demo_2 --models-dir models/pd_mlp_demo_2`.

### Comandos y protocolo posterior

Windows PowerShell, desde la raíz, para repetir la búsqueda, las corridas finales y la comparación en salidas nuevas (no sobrescribe los directorios ya existentes):

```powershell
python scripts/14_entrenar_mlp_pd.py --mode search --output-dir results/pd_mlp_search_reproduccion --models-dir models/pd_mlp_search_reproduccion
python scripts/14_entrenar_mlp_pd.py --mode final --search-dir results/pd_mlp_search_reproduccion --output-dir results/pd_mlp_final_reproduccion --models-dir models/pd_mlp_final_reproduccion
python scripts/11_tabla_comparacion_pd.py --mlp results/pd_mlp_final_reproduccion --salida results/pd_comparacion_mlp_reproduccion --figuras figures/pd_mlp_comparacion_reproduccion
```

Linux, desde la raíz, después de activar `.venv`:

```bash
python scripts/14_entrenar_mlp_pd.py --mode search --output-dir results/pd_mlp_search_reproduccion --models-dir models/pd_mlp_search_reproduccion
python scripts/14_entrenar_mlp_pd.py --mode final --search-dir results/pd_mlp_search_reproduccion --output-dir results/pd_mlp_final_reproduccion --models-dir models/pd_mlp_final_reproduccion
python scripts/11_tabla_comparacion_pd.py --mlp results/pd_mlp_final_reproduccion --salida results/pd_comparacion_mlp_reproduccion --figuras figures/pd_mlp_comparacion_reproduccion
```

La búsqueda exploratoria produce tabla de métricas de validación por configuración/semilla, curvas de calibración de validación, figura de log-loss y un manifiesto con hashes. `--mode final` exige una búsqueda con selección independiente por partición y los mismos hashes de datos, preprocesador, logística, configuración y código. Entrena al menos 30 semillas por partición, evalúa validación/prueba con el procedimiento ya fijado y guarda checkpoints que incluyen la configuración propia de su partición. El comparador resume AUC, Gini, KS, Brier y log-loss con media, desviación, mejor y peor; las curvas de prueba por semilla respaldan calibración. En el caso ejecutado, el manifiesto de búsqueda habilita la fase final (`seleccion_final_habilitada=true`); el manifiesto final pone ese indicador en `false` para evitar reutilizar una búsqueda ya consumida. Los datos, el preprocesador actual, la logística y el código se verifican mediante sus hashes.

### Resultados de búsqueda y decisión de configuración

La búsqueda ejecutada recorrió las 12 configuraciones, semillas 2026/2027 y ambas particiones: 48 ajustes. Guardó 48 checkpoints, 48 registros métricos de validación y 480 filas de calibración (10 deciles por ajuste). Tiempo sumado de entrenamiento: **1.602,3 s** (aleatoria 835,8 s; temporal 766,5 s); tiempo total de ejecución registrado: **1.606,9 s**. El proceso completó dentro del presupuesto de 60 minutos. El criterio para elegir fue la menor media de log-loss de validación de las dos semillas, aplicada por separado.

Las celdas de métricas muestran medias de las dos semillas exploratorias. `ECE decil` es un diagnóstico descriptivo: promedio por semilla del error absoluto entre tasa observada y PD media, ponderado por tamaño de decil; no participó en la selección. Las diferencias negativas de log-loss frente a la base favorecen al candidato bajo el criterio predeclarado. El “costo” es la suma de segundos de ajuste de sus dos semillas en esa partición.

**Partición aleatoria**

| Configuración | Log-loss (Δ base) | AUC / Gini / KS / Brier | ECE decil | Épocas media | Segundos | Lectura según criterio |
|---|---:|---|---:|---:|---:|---|
| `base_32_relu_adam` | 0,479078 (base) | 0,706511 / 0,413021 / 0,304682 / 0,155232 | 0,006593 | 16,0 | 63,4 | Referencia. |
| `width64` | **0,478570 (−0,000508)** | 0,707709 / 0,415419 / 0,309441 / 0,155064 | 0,006276 | 16,0 | 63,3 | Mejor log-loss; mejora pequeña, seleccionada. |
| `depth32_16` | 0,479034 (−0,000044) | 0,706834 / 0,413668 / 0,306163 / 0,155233 | 0,007341 | 16,0 | 61,5 | Prácticamente igual a base. |
| `activation_tanh` | 0,482503 (+0,003425) | 0,699799 / 0,399597 / 0,291465 / 0,156479 | 0,009428 | 16,0 | 63,3 | Peor que base. |
| `dropout_020` | 0,479924 (+0,000846) | 0,704554 / 0,409107 / 0,300720 / 0,155487 | 0,004348 | 16,0 | 67,8 | Peor log-loss; ECE menor, trade-off descriptivo. |
| `weight_decay_1e4` | 0,479085 (+0,000007) | 0,706475 / 0,412951 / 0,305152 / 0,155233 | 0,006622 | 16,0 | 74,9 | Efecto casi nulo. |
| `optimizer_adamw` | 0,479040 (−0,000038) | 0,706566 / 0,413132 / 0,305448 / 0,155216 | 0,006921 | 16,0 | 73,4 | Casi igual; comparada con Adam a igual WD. |
| `optimizer_sgd` | 0,495630 (+0,016553) | 0,673097 / 0,346195 / 0,256043 / 0,160858 | 0,022990 | 16,0 | 72,8 | No útil bajo esta tasa y este presupuesto. |
| `learning_rate_0003` | 0,482469 (+0,003391) | 0,699391 / 0,398783 / 0,293078 / 0,156411 | 0,006546 | 16,0 | 73,7 | Peor que base. |
| `learning_rate_003` | 0,478664 (−0,000413) | 0,707450 / 0,414900 / 0,307089 / 0,155098 | 0,008267 | 14,5 | 66,8 | Segunda por log-loss; cerca de `width64`. |
| `combo_width64_dropout020` | 0,478661 (−0,000417) | 0,707127 / 0,414253 / 0,305185 / 0,155051 | 0,004849 | 16,0 | 80,5 | Mejora pequeña; interacción, no atribuir a anchura o dropout por separado. |
| `combo_depth_weight_decay` | 0,478931 (−0,000147) | 0,707079 / 0,414158 / 0,306451 / 0,155212 | 0,006958 | 16,0 | 74,4 | Mejora muy pequeña; interacción. |

**Partición temporal**

| Configuración | Log-loss (Δ base) | AUC / Gini / KS / Brier | ECE decil | Épocas media | Segundos | Lectura según criterio |
|---|---:|---|---:|---:|---:|---|
| `base_32_relu_adam` | 0,530856 (base) | 0,696556 / 0,393112 / 0,287511 / 0,176539 | 0,033148 | 16,0 | 64,2 | Referencia. |
| `width64` | **0,530349 (−0,000507)** | 0,696872 / 0,393743 / 0,287820 / 0,176380 | 0,031286 | 16,0 | 65,3 | Mejor log-loss; mejora pequeña, seleccionada. |
| `depth32_16` | 0,531720 (+0,000864) | 0,694740 / 0,389479 / 0,283937 / 0,176877 | 0,032700 | 16,0 | 62,8 | Peor que base. |
| `activation_tanh` | 0,534050 (+0,003194) | 0,690740 / 0,381479 / 0,279389 / 0,177727 | 0,032877 | 16,0 | 66,0 | Peor que base. |
| `dropout_020` | 0,531749 (+0,000893) | 0,694286 / 0,388572 / 0,284386 / 0,176887 | 0,032415 | 16,0 | 66,2 | Peor log-loss, aunque el gap de decil baja levemente. |
| `weight_decay_1e4` | 0,530674 (−0,000182) | 0,696702 / 0,393405 / 0,288145 / 0,176495 | 0,032586 | 16,0 | 64,0 | Mejora pequeña. |
| `optimizer_adamw` | 0,530824 (−0,000032) | 0,696536 / 0,393071 / 0,287343 / 0,176525 | 0,032994 | 16,0 | 64,3 | Casi igual a base/Adam con WD. |
| `optimizer_sgd` | 0,554126 (+0,023270) | 0,658727 / 0,317454 / 0,234744 / 0,185726 | 0,048376 | 16,0 | 64,1 | No útil bajo esta tasa y este presupuesto. |
| `learning_rate_0003` | 0,536995 (+0,006140) | 0,686192 / 0,372385 / 0,272085 / 0,178786 | 0,036790 | 16,0 | 64,0 | Peor que base. |
| `learning_rate_003` | 0,530453 (−0,000403) | 0,696096 / 0,392192 / 0,286791 / 0,176399 | 0,028730 | 12,5 | 49,6 | Segunda por log-loss; menos épocas, ECE menor. |
| `combo_width64_dropout020` | 0,530633 (−0,000223) | 0,696797 / 0,393594 / 0,288165 / 0,176467 | 0,032709 | 16,0 | 70,0 | Mejora pequeña, inferior a `width64`; interacción. |
| `combo_depth_weight_decay` | 0,531135 (+0,000279) | 0,695485 / 0,390971 / 0,284701 / 0,176735 | 0,031880 | 16,0 | 65,9 | Peor log-loss que base; interacción. |

La anchura 64 quedó elegida en ambas particiones por el criterio predeclarado. Su ventaja media de log-loss frente a base es de alrededor de **0,00051**, pequeña y observada con solo dos semillas exploratorias. El LR 0,003 y una combinación quedaron muy cerca en aleatoria; LR 0,003 quedó segundo temporal. La elección no prueba superioridad amplia: refleja la regla acordada y se cuantifica mejor con 30 semillas. No se atribuye el resultado de las combinaciones a un factor aislado.

Los detalles por semilla, parámetros JSON exactos, épocas, métricas, duración, versiones y hashes están en `results/pd_mlp_search_independiente_v1/pd_mlp_busqueda.csv`. Las curvas de calibración por candidato están en `curva_calibracion_busqueda.csv`; `figura_busqueda_pd_mlp.png` muestra log-loss de validación con barras de dispersión entre las dos semillas. Se guardaron los checkpoints exploratorios en `models/pd_mlp_search_independiente_v1/`.

### Corridas finales de 30 semillas

Con las configuraciones ya fijadas (`width64`: 99→64→1, ReLU, Adam, lr 0,001, sin dropout ni weight decay), el modo final ajustó 30 semillas (0–29) para cada partición: **60 modelos**, 120 filas (30 por partición y conjunto de validación/prueba) y 1.200 filas de calibración (10 deciles × 30 semillas × 2 conjuntos × 2 particiones). Tiempo sumado de ajuste: **1.845,7 s**; ejecución total: **1.854,5 s**. El número de parámetros es 6.465 por modelo. Todas las métricas finitas y 30 semillas únicas por cada combinación se verificaron. La prueba se midió una vez terminado el protocolo; no se hicieron ajustes posteriores.

Media, desviación estándar muestral, mejor y peor de cada métrica, sobre las 30 semillas:

| Partición | Conjunto | Métrica | Media | Desv. est. | Mejor | Peor |
|---|---|---|---:|---:|---:|---:|
| Aleatoria | Validación | AUC | 0,707622 | 0,000581 | 0,708958 | 0,706460 |
| Aleatoria | Validación | Gini | 0,415243 | 0,001161 | 0,417917 | 0,412921 |
| Aleatoria | Validación | KS | 0,307661 | 0,001854 | 0,312884 | 0,304947 |
| Aleatoria | Validación | Brier | 0,155073 | 0,000101 | 0,154893 | 0,155292 |
| Aleatoria | Validación | Log-loss | 0,478605 | 0,000282 | 0,477955 | 0,479237 |
| Aleatoria | Prueba | AUC | 0,706664 | 0,000503 | 0,707840 | 0,705736 |
| Aleatoria | Prueba | Gini | 0,413328 | 0,001005 | 0,415680 | 0,411471 |
| Aleatoria | Prueba | KS | 0,303731 | 0,001692 | 0,307271 | 0,299733 |
| Aleatoria | Prueba | Brier | 0,154747 | 0,000084 | 0,154581 | 0,154933 |
| Aleatoria | Prueba | Log-loss | 0,477987 | 0,000227 | 0,477552 | 0,478438 |
| Temporal | Validación | AUC | 0,696444 | 0,000979 | 0,698820 | 0,694017 |
| Temporal | Validación | Gini | 0,392887 | 0,001958 | 0,397639 | 0,388034 |
| Temporal | Validación | KS | 0,286275 | 0,001775 | 0,290485 | 0,282564 |
| Temporal | Validación | Brier | 0,176527 | 0,000240 | 0,176148 | 0,177065 |
| Temporal | Validación | Log-loss | 0,530803 | 0,000630 | 0,529875 | 0,532213 |
| Temporal | Prueba | AUC | 0,686268 | 0,001621 | 0,690224 | 0,682815 |
| Temporal | Prueba | Gini | 0,372535 | 0,003243 | 0,380447 | 0,365630 |
| Temporal | Prueba | KS | 0,274678 | 0,003898 | 0,281982 | 0,267245 |
| Temporal | Prueba | Brier | 0,175400 | 0,000303 | 0,174751 | 0,175973 |
| Temporal | Prueba | Log-loss | 0,527516 | 0,000772 | 0,525591 | 0,528623 |

En la prueba, frente a la logística ya guardada con estos mismos artefactos, el promedio MLP fue mejor en las cinco métricas: aleatoria AUC 0,7067 vs 0,6918, Brier 0,1547 vs 0,1572 y log-loss 0,4780 vs 0,4859; temporal AUC 0,6863 vs 0,6749, Brier 0,1754 vs 0,1773 y log-loss 0,5275 vs 0,5360. Son comparaciones descriptivas de los artefactos aceptados. La lectura temporal conserva la limitación del preprocesador actual y no debe presentarse como evidencia libre de fuga.

Como chequeo descriptivo de calibración, el error absoluto medio ponderado de decil en prueba promedió 0,00715 (DE entre semillas 0,00105) para aleatoria y 0,02502 (0,00210) para temporal. La logística correspondiente dio 0,00498 y 0,03164. Este resumen de deciles complementa Brier y log-loss; la curva completa se encuentra en las figuras y los CSV de calibración. El MLP temporal mejora ese gap frente a la logística observada, pero la limitación del preprocesador impide una conclusión metodológica fuerte.

Artefactos finales: `results/pd_mlp_final_particionado/` (métricas por semilla, curvas de calibración, resumen/hash, figura de calibración y checkpoints en `models/pd_mlp_final_particionado/`). La tabla comparativa de AUC/Gini/KS/Brier/log-loss quedó en `results/pd_comparacion_mlp_final_v2/tabla_comparacion_pd.csv`; la figura logística/MLP en prueba quedó en `figures/pd_mlp_final_comparison_v2/fig09_calibracion_comparada.png`. Los resultados originales no se sobrescribieron.

La demo anterior tomó 14,7 s con dos épocas; sus cifras de prueba son solo evidencia de cableado y **no fueron usadas para selección**. Resultados temporales de la búsqueda y semillas finales usan los artefactos aceptados actualmente y mantienen la limitación de que el preprocesador se ajustó con entrenamiento aleatorio, que puede incluir filas de particiones temporales. Si A cambia el ajuste y regenera preprocesador/logística, los resultados temporales deben repetirse. La decisión actual del equipo permite ejecutar el plan sin esperar esa regeneración, sin presentar el resultado como libre de fuga.

## Sensibilidad posterior: ponderación de clases en PD

Este experimento conserva como referencia el `width64` sin ponderación ya elegido por validación y entrenado con 30 semillas. No reemplaza sus modelos ni resultados. La decisión del equipo de conservar el preprocesador sigue siendo información comunicada; la evidencia de A continúa pendiente. El estudio temporal hereda la limitación descrita arriba y no es una nueva selección final imparcial.

### Protocolo fijado antes de ejecutar

| Condición | Peso positivo | Hipótesis previa | Interpretación esperada |
|---|---:|---|---|
| `control_pos_weight_1` | 1 | Control del ajuste sin ponderación; debe conservar la escala de probabilidad del modelo principal. | Referencia pareada para log-loss, Brier y calibración. |
| `moderate_pos_weight_2` | 2 | Dar peso doble a positivos puede elevar sensibilidad/discriminación con un cambio moderado de escala de PD. | Solo sería candidata si mejora calidad probabilística en validación y no degrada materialmente calibración. |
| `train_ratio_pos_weight` | n0/n1 de entrenamiento | Compensación completa de la frecuencia de la clase 1 puede mejorar el ordenamiento, pero desplazar las probabilidades fuera de la prevalencia natural. | Contraste más fuerte; una AUC/KS mayor no basta para recomendarla como PD. |

Antes del entrenamiento, el parquet dio estos pesos, calculados solo con las etiquetas de `entrenamiento` de cada partición:

| Partición | n clase 0 train | n clase 1 train | `n0/n1` | Prevalencia clase 1 train | Prevalencia validación | Prevalencia prueba |
|---|---:|---:|---:|---:|---:|---:|
| Aleatoria | 146.797 | 41.173 | 3,565371 | 0,219040 | 0,219042 | 0,219042 |
| Temporal | 132.014 | 31.527 | 4,187331 | 0,192777 | 0,262171 | 0,254613 |

Se fijaron semillas pareadas **41, 42 y 43** para las tres condiciones y ambas particiones: 3 × 3 × 2 = **18 ajustes**, mismos 99 predictores transformados, arquitectura 99→64→1 ReLU, Adam, tasa 0,001, batch 2.048, máximo 16 épocas, paciencia 4 y `min_delta=0,0001`. Se cambia exclusivamente `pos_weight` de la pérdida de entrenamiento. No hay sobremuestreo ni submuestreo. La parada temprana conserva el criterio de log-loss **sin ponderar** en la validación de esa partición, para comparar PD bajo la prevalencia natural. Presupuesto declarado: **20 minutos CPU**; se registran tiempos por corrida y se comprueba el total.

La comparación primaria se limita a las métricas de validación, medias/desviaciones entre las tres semillas y diferencias pareadas por semilla frente a `pos_weight=1`. Se consideran log-loss y Brier junto con calibración y la diferencia entre PD media predicha y tasa observada; AUC/Gini/KS describen ordenamiento, pero no deciden solos. La prueba se calcula y etiqueta aparte únicamente como **análisis de sensibilidad**, porque ya se había inspeccionado en el desarrollo del modelo principal; no se usa para escoger peso, recalibrar ni reemplazar la evaluación ya entregada. La tasa de positivos permanece intacta en validación y prueba.

La función de entrenamiento añadió un argumento opcional `pos_weight` con valor por defecto 1; así los entrenamientos existentes conservan su objetivo. Para una etiqueta Bernoulli, PyTorch aplica `- [w y log σ(z) + (1-y) log(1-σ(z))]` a cada fila de train. Si `w>1`, errores en positivos pesan más; el modelo puede compensar elevando logits y mover la probabilidad media aunque la prevalencia de evaluación no cambie. Por eso el peso de clases no equivale a una calibración posterior y se revisan explícitamente Brier, log-loss y calibración.

### Resultados observados de sensibilidad

Se ejecutó el protocolo anterior con `.venv`: **18 ajustes**, 411,685 s sumados de entrenamiento y 416,121 s de pared, bajo el presupuesto de 20 minutos. Hubo 36 registros métricos (validación/prueba × 18 modelos) y 360 filas de calibración. La tabla siguiente usa solo validación para comparar condiciones; muestra media entre tres semillas, desviación muestral de log-loss, métricas de ordenamiento y probabilidades. `ECE decil` es el error absoluto medio de calibración ponderado por tamaño de decil. El tiempo es la suma de los tres ajustes de cada condición y partición.

| Partición | Condición (peso) | Log-loss media (DE) | AUC / Gini / KS | Brier | ECE decil | PD media vs tasa | Épocas medias | Segundos (3 semillas) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Aleatoria | Control (1) | 0,478469 (0,000059) | 0,707994 / 0,415987 / 0,308819 | 0,155014 | 0,005405 | 0,222427 vs 0,219042 | 16,00 | 94,01 |
| Aleatoria | Moderado (2) | 0,514983 (0,003328) | 0,706138 / 0,412275 / 0,304494 | 0,169711 | 0,114472 | 0,333514 vs 0,219042 | 11,67 | 68,38 |
| Aleatoria | Cociente train (3,565371) | 0,617959 (0,007959) | 0,703367 / 0,406734 / 0,299032 | 0,214830 | 0,236467 | 0,455509 vs 0,219042 | 10,00 | 58,81 |
| Temporal | Control (1) | 0,530953 (0,000588) | 0,696068 / 0,392136 / 0,286089 | 0,176529 | 0,031805 | 0,230365 vs 0,262171 | 15,00 | 76,38 |
| Temporal | Moderado (2) | 0,544965 (0,001972) | 0,696716 / 0,393432 / 0,287106 | 0,182743 | 0,081743 | 0,343914 vs 0,262171 | 13,00 | 65,97 |
| Temporal | Cociente train (4,187331) | 0,660383 (0,006392) | 0,694136 / 0,388271 / 0,284076 | 0,233999 | 0,236155 | 0,498326 vs 0,262171 | 9,33 | 48,13 |

Las diferencias pareadas (tratamiento menos control, mismas semillas) son más claras que una comparación de corridas distintas: en validación aleatoria, peso 2 aumentó log-loss **0,036515** y Brier **0,014697**, con AUC −0,001856; en temporal, aumentó log-loss **0,014012** y Brier **0,006214**, mientras AUC aumentó solo 0,000648. El cociente train elevó log-loss en **0,139490** y **0,129430**, respectivamente, y Brier en **0,059816** y **0,057470**. Los dos tratamientos elevaron mucho la PD media y alejaron la calibración de la tasa observada; el cociente completo fue el peor en calidad probabilística. No hay evidencia para usar ponderación como PD en este protocolo. El menor tiempo de los tratamientos se explica en parte por menos épocas hasta la parada temprana, pero no compensa el deterioro.

La prueba se presenta únicamente como sensibilidad posterior: no cambió ninguna elección. En prueba, log-loss/Brier/AUC promedio fueron: aleatoria control 0,478158/0,154831/0,706304, peso 2 0,514107/0,169348/0,705440, cociente 0,617079/0,214571/0,702939; temporal control 0,528058/0,175550/0,686017, peso 2 0,551853/0,186229/0,686884 y cociente 0,677270/0,241233/0,685202. Estas cifras no constituyen una nueva evaluación final independiente y la lectura temporal conserva la limitación del preprocesador.

Las salidas se guardaron sin sobrescribir el modelo principal: `results/pd_mlp_ponderacion_sensibilidad_v1/` y `models/pd_mlp_ponderacion_sensibilidad_v1/`. `metricas_ponderacion_por_semilla.csv` registra cada condición, partición y semilla; `curva_calibracion_ponderacion.csv` contiene deciles; `resumen_ponderacion_media_desviacion.csv` y `resumen_diferencias_pareadas_vs_control*.csv` permiten reproducir las tablas. `figura_logloss_validacion.png` y `figura_calibracion_validacion.png` excluyen prueba. Se guardaron 18 checkpoints. `resumen_ponderacion.json` preserva versiones, hashes, pesos calculados solo con train y limitación del preprocesador.

**Límite de decisión:** usar validación para ordenar estos tres tratamientos hace de este resultado una sensibilidad posterior con solo tres semillas, no una nueva selección imparcial. Las cifras de prueba no justifican selección ni sustituyen las 30 corridas principales. Si un peso parece útil, debe quedar como candidato para una evaluación futura independiente, con nuevos datos/particiones o protocolo acordado; no reemplaza automáticamente el `width64` principal.

Reproducción (usa carpetas nuevas para otra ejecución):

```powershell
python scripts/15_experimento_ponderacion_pd.py
```

```bash
python scripts/15_experimento_ponderacion_pd.py
```

### Alcance de los tiempos en la comparación PD

`scripts/11_tabla_comparacion_pd.py` distingue búsqueda por partición, tiempo de corridas finales registrado globalmente y total global calculable. Los JSON originales confirman búsqueda MLP **1.602,258356 s** (aleatoria 835,801914; temporal 766,456442), ajuste final MLP **1.845,709787 s** y suma de entrenamiento búsqueda+final **3.447,968143 s**. El manifiesto no desglosa el ajuste final por partición: la tabla deja vacías las celdas finales y el total atribuible a cada partición; no reparte el valor agregado. El total global no incluye overhead entre procesos. Para la logística se registró búsqueda de 19,120 s aleatoria y 16,339 s temporal (35,459 s total); no hay un tiempo separado de una corrida final.
