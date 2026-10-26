# Trabajo 2: Redes neuronales para datos tabulares

## Entorno reproducible

El entorno se fija con **Python 3.12.14** y las versiones de `requirements.txt`. Los resultados y modelos existentes registran Python 3.10.12 y NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, pyarrow 25.0.1 y joblib 1.6.0. Se conserva ese conjunto de bibliotecas para compatibilidad inicial con los artefactos. La versión 3.12.14 es la disponible en el equipo de desarrollo para este entorno nuevo; los archivos `.joblib` se cargaron y probaron desde él.

PyTorch está fijado a la distribución **CPU** (`2.8.0+cpu`). El equipo de desarrollo observado usa un AMD Ryzen 7 5700G y no se encontró evidencia de una GPU NVIDIA/CUDA configurada. La instalación no depende de CUDA ni de ROCm. Si el equipo decide usar GPU, debe acordar y documentar otra distribución antes de cambiar el entorno.

La integración de LGD y pérdida esperada usa NumPy, pandas, SciPy, scikit-learn, pyarrow, joblib y matplotlib, ya incluidos en este conjunto fijo; no requiere paquetes adicionales. Se conservan las versiones verificadas con los artefactos existentes en lugar de cambiar el entorno al resolver el `requirements.txt` de ambos frentes.

No hay todavía código de aplicación web, por lo que no se agregan dependencias de interfaz. `pytest` tampoco es necesario para la prueba actual: `tests/test_scorecard.py` puede ejecutarse directamente con Python.

### Windows (PowerShell)

Requiere Python 3.12.14 disponible como `py -3.12`. En el host auditado, `py -0p` solo lista Python 3.14; la `.venv` actual se creó con Python 3.12.14 del runtime local de Codex. Para recrearla en ese host usa el comando exacto registrado en `docs/entorno_y_balance_pd.md`, o instala Python 3.12.14 para que `py -3.12` esté disponible.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python --version
python -m pip --version
python -m pip freeze
python tests/test_scorecard.py
python scripts/13_balance_clases_pd.py
```

Si la política de PowerShell impide activar el entorno, llama a su ejecutable directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tests/test_scorecard.py
.\.venv\Scripts\python.exe scripts/13_balance_clases_pd.py
```

### Linux

Requiere Python 3.12.14 y `venv`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python --version
python -m pip --version
python -m pip freeze
python tests/test_scorecard.py
python scripts/13_balance_clases_pd.py
```

### Comprobación de Git

La carpeta `.venv` está excluida con una regla anclada a la raíz (`/.venv/`). Compruébalo desde la raíz:

```bash
git check-ignore -v .venv/pyvenv.cfg
git status --short --ignored
```

En Windows PowerShell se pueden usar los mismos comandos `git`.

## Datos y comandos existentes

El archivo original `data/raw/loan.csv` no está versionado porque supera el límite de GitHub. `data/raw/README.md` describe cómo obtenerlo de Kaggle. Los experimentos existentes consumen `data/processed/pd.parquet`; el diccionario `data/raw/LCDataDictionary.xlsx` sí está incluido.

El balance de clases se comprueba sin modificar archivos con:

```bash
python scripts/13_balance_clases_pd.py
```

`python tests/test_scorecard.py` también es una comprobación de solo lectura. En cambio, `python scripts/06_generar_datos_pd.py`, `scripts/08_regresion_logistica.py`, `scripts/09_guardar_logistica.py` y `scripts/10_scorecard.py` generan o sobrescriben artefactos en `data/processed/`, `models/`, `results/` o `figures/`; ejecútalos solo cuando se haya acordado regenerar esos resultados. La búsqueda logística necesita el CSV original. El script de logística registra versiones y resultados, pero su lectura de hardware se ha hecho portable para Windows y Linux.

## Estado del balance PD

La partición aleatoria usa estratificación para mantener aproximadamente la prevalencia global en entrenamiento, validación y prueba. La partición temporal conserva los años de originación en su orden y, por tanto, muestra cambios reales de prevalencia entre periodos. La estratificación no crea más ejemplos de la clase minoritaria ni cambia el conjunto; balancear sí modifica la frecuencia efectiva durante el ajuste, mediante ponderación o remuestreo.

El script de balance reporta ambas clases en cada partición. La validación y prueba se mantienen con la distribución natural. No se activa ponderación ni remuestreo automáticamente. Una comparación de balance debe ajustar pesos o remuestreo exclusivamente dentro de entrenamiento y evaluar tanto tratamiento como control en la misma validación y prueba no modificadas, comparando AUC, KS, Brier y calibración.

## Decisiones y verificación del entorno

Las versiones fijadas, comandos ejecutados, prueba de scorecard, lectura del parquet, carga de modelos y balance de clases quedan documentados en [`docs/entorno_y_balance_pd.md`](docs/entorno_y_balance_pd.md). Los resultados no deben regenerarse solo para hacer pasar una validación.

## MLP de probabilidad de incumplimiento (Persona B)

El protocolo, sus límites, la decisión comunicada sobre el preprocesador y los comandos están en [`docs/entorno_y_balance_pd.md`](docs/entorno_y_balance_pd.md). La demo carga el mismo parquet/preprocesador que la logística y verifica ejecución e integración; no desempeño final. La búsqueda selecciona por separado dentro de cada partición y no usa prueba. Se conserva el preprocesador actual por decisión comunicada del equipo; la limitación temporal permanece documentada, así que sus resultados temporales deben repetirse si A cambia el ajuste y regenera la logística. La búsqueda independiente y las 30 semillas por partición ya se ejecutaron sobre los artefactos actuales; siguen siendo provisionales para interpretación temporal por la limitación del preprocesador.

```powershell
.\.venv\Scripts\python.exe scripts/14_entrenar_mlp_pd.py --mode demo
```

```bash
python scripts/14_entrenar_mlp_pd.py --mode demo
```

La búsqueda con `--mode search` selecciona independientemente una configuración por partición y usa solo la validación de esa partición; las configuraciones pueden coincidir, como ocurrió en esta ejecución. El modo `--mode final` conserva hashes del parquet, preprocesador, modelos logísticos, protocolo, configuración y código para comprobar que la búsqueda sigue correspondiendo a los artefactos evaluados. Los resultados temporales conservan la limitación conocida del ajuste vigente del preprocesador; la decisión del equipo de mantenerlo no demuestra ausencia de fuga.

La comparación PD reporta por separado búsqueda, corridas finales globales y sumas calculables; el tiempo de entrenamiento final del MLP no está desglosado por partición. Una sensibilidad posterior con `pos_weight` (control 1, peso 2 y cociente de clases de train) está documentada en `docs/entorno_y_balance_pd.md`; no reemplaza el modelo principal y sus pruebas no se usan para elegir pesos.

Reproducción en carpetas de salida nuevas (el script no sobrescribe resultados existentes):

```powershell
python scripts/14_entrenar_mlp_pd.py --mode search --output-dir results/pd_mlp_search_reproduccion --models-dir models/pd_mlp_search_reproduccion
python scripts/14_entrenar_mlp_pd.py --mode final --search-dir results/pd_mlp_search_reproduccion --output-dir results/pd_mlp_final_reproduccion --models-dir models/pd_mlp_final_reproduccion
python scripts/11_tabla_comparacion_pd.py --mlp results/pd_mlp_final_reproduccion --salida results/pd_comparacion_mlp_reproduccion --figuras figures/pd_mlp_comparacion_reproduccion
```

En Linux se ejecutan los mismos comandos después de activar `.venv`.

## MLP de LGD (Persona B)

La auditoría de la muestra, sus límites, el baseline Ridge y el protocolo futuro están en [`docs/mlp_lgd.md`](docs/mlp_lgd.md). El demo configurable de dos épocas usa la misma muestra castigada, transformación y máscaras de Ridge; evalúa validación solamente y no es resultado final.

El piloto de convergencia (dos inicializaciones × dos semillas × dos particiones, validación únicamente) y su extensión están en [`docs/mlp_lgd_convergencia.md`](docs/mlp_lgd_convergencia.md). Los resultados canónicos corregidos quedan en `results/lgd_mlp_convergencia_piloto_v2/` y `results/lgd_mlp_convergencia_extension_v2/`; no sustituyen la búsqueda ni las corridas finales.

El protocolo de diez configuraciones exploratorias de LGD y su comando están en [`docs/mlp_lgd_busqueda.md`](docs/mlp_lgd_busqueda.md). La búsqueda selecciona por separado en cada partición y no calcula métricas de prueba.

```powershell
.\.venv\Scripts\python.exe scripts/19_buscar_mlp_lgd.py --modo search
```

La extensión acotada de las dos semillas de inicialización PyTorch se repite con:

```powershell
.\.venv\Scripts\python.exe scripts/20_extender_busqueda_mlp_lgd.py
```

La fase final congelada de 30 semillas por partición y su análisis están en [`docs/mlp_lgd_final.md`](docs/mlp_lgd_final.md); se verifica el preflight antes del entrenamiento y la prueba se evalúa después de guardar los 60 checkpoints.

```powershell
.\.venv\Scripts\python.exe scripts/16_entrenar_mlp_lgd.py
```

```bash
python scripts/16_entrenar_mlp_lgd.py
```
