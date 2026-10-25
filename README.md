# Trabajo 2: Redes neuronales para datos tabulares

## Entorno reproducible

El entorno se fija con **Python 3.12.14** y las versiones de `requirements.txt`. Los resultados y modelos existentes registran Python 3.10.12 y NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, pyarrow 25.0.1 y joblib 1.6.0. Se conserva ese conjunto de bibliotecas para compatibilidad inicial con los artefactos. La versión 3.12.14 es la disponible en el equipo de desarrollo para este entorno nuevo; los archivos `.joblib` se cargaron y probaron desde él.

PyTorch está fijado a la distribución **CPU** (`2.8.0+cpu`). El equipo de desarrollo observado usa un AMD Ryzen 7 5700G y no se encontró evidencia de una GPU NVIDIA/CUDA configurada. La instalación no depende de CUDA ni de ROCm. Si el equipo decide usar GPU, debe acordar y documentar otra distribución antes de cambiar el entorno.

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

El protocolo, sus límites, la corrida demostrativa observada y los comandos están en [`docs/entorno_y_balance_pd.md`](docs/entorno_y_balance_pd.md). La demo carga el mismo `data/processed/pd.parquet` y `models/preprocessor.joblib` que la logística, usa `feature_names_in_` y las máscaras `split_random`/`split_temporal`, y guarda salidas nuevas bajo `results/pd_mlp_demo/` y `models/pd_mlp_demo/`. Verifica ejecución e integración, no desempeño final. La búsqueda global actual tiene solapamiento entre validaciones y pruebas de las dos particiones; no se debe usar para seleccionar la arquitectura final. Los resultados temporales deben repetirse cuando A corrija el ajuste del preprocesador y regenere la logística. No se han ejecutado la búsqueda válida ni las 30 semillas finales.

```powershell
.\.venv\Scripts\python.exe scripts/14_entrenar_mlp_pd.py --mode demo
```

```bash
python scripts/14_entrenar_mlp_pd.py --mode demo
```

La búsqueda pequeña se ejecuta con `--mode search`; la corrida definitiva de 30 semillas está bloqueada hasta que el frente de datos regenere preprocesador, datos y logística coherentes, y solo entonces se confirme explícitamente el protocolo.
