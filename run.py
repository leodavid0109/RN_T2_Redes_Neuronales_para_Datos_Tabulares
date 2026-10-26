"""Lanzador de experimentos: un comando por experimento.

Uso (desde la raíz del repo, con el entorno activo):
    python run.py                      lista los comandos
    python run.py <comando> [args]     ejecuta un comando; los args extra pasan al script

Los args extra se pasan tal cual al script, por ejemplo:
    python run.py lgd-lineal --alphas 0.1 1 10 100
    python run.py perdida-esperada --particion temporal
"""
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# nombre -> (descripción, [scripts en orden], ¿acepta args extra?)
COMANDOS = {
    "datos": ("Genera pd.parquet y lgd.parquet desde data/raw/loan.csv (requiere el CSV original)",
              ["scripts/06_generar_datos_pd.py", "src/lgd.py"], False),
    "descriptivo": ("Tablas y figuras del análisis descriptivo e hipótesis (Parte 1)",
                    ["scripts/04_tablas_figuras_datos.py", "scripts/07_analisis_descriptivo.py"], False),
    "pd-logreg": ("Regresión logística de referencia (PD): búsqueda y modelos guardados",
                  ["scripts/08_regresion_logistica.py", "scripts/09_guardar_logistica.py"], False),
    "scorecard": ("Scorecard de la PD (parámetros en configs/scorecard.yaml)",
                  ["scripts/10_scorecard.py"], False),
    "tablas-pd": ("Tabla de comparación justa de PD y tabla final de hipótesis",
                  ["scripts/11_tabla_comparacion_pd.py", "scripts/12_tabla_hipotesis_final.py"], False),
    "lgd-lineal": ("LGD: Ridge de referencia (admite --alphas, --demo, --seed)",
                   ["src/lgd_lineal.py"], True),
    "lgd-figura": ("LGD: figura de distribución predicha vs. observada",
                   ["scripts/13_figura_lgd_lineal.py"], False),
    "perdida-esperada": ("Pérdida esperada E[L] = PD·LGD·EAD (admite --particion, --conjunto)",
                         ["src/expected_loss.py"], True),
    "tests": ("Pruebas de la scorecard", ["tests/test_scorecard.py"], False),
}

# Combinaciones que no necesitan el CSV original
COMBOS = {
    "demo": ("Demostración (< 2 min): LGD en modo demo + pérdida esperada + pruebas",
             [("lgd-lineal", ["--demo"]), ("perdida-esperada", []), ("tests", [])]),
    "todo": ("Todo lo que no requiere el CSV original",
             [("pd-logreg", []), ("scorecard", []), ("lgd-lineal", []), ("lgd-figura", []),
              ("perdida-esperada", ["--particion", "aleatoria"]),
              ("perdida-esperada", ["--particion", "temporal"]), ("tablas-pd", []), ("tests", [])]),
}


def correr(script, extra):
    cmd = [sys.executable, str(RAIZ / script), *extra]
    print(f"\n>>> {' '.join(['python', script, *extra])}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=RAIZ)
    print(f"<<< {script}: {time.perf_counter() - t0:.1f} s", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def ejecutar(nombre, extra):
    desc, scripts, acepta = COMANDOS[nombre]
    if extra and not acepta:
        print(f"Aviso: '{nombre}' no admite argumentos extra; se ignoran: {extra}")
        extra = []
    for i, s in enumerate(scripts):
        # los args extra solo van al último script del comando
        correr(s, extra if i == len(scripts) - 1 else [])


def app():
    ruta = RAIZ / "app" / "app.py"
    if not ruta.exists():
        sys.exit("La aplicación todavía no existe (falta app/app.py).")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ruta)], cwd=RAIZ)


def ayuda():
    print(__doc__)
    print("Comandos:")
    for k, (d, *_ ) in {**COMANDOS, **COMBOS}.items():
        print(f"  {k:<18} {d}")
    print(f"  {'app':<18} Levanta la aplicación web local (Streamlit)")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "ayuda"):
        return ayuda()
    nombre, extra = sys.argv[1], sys.argv[2:]
    t0 = time.perf_counter()
    if nombre in COMANDOS:
        ejecutar(nombre, extra)
    elif nombre in COMBOS:
        for sub, args in COMBOS[nombre][1]:
            ejecutar(sub, args)
    elif nombre == "app":
        return app()
    else:
        print(f"Comando desconocido: {nombre}\n")
        return ayuda()
    print(f"\nTiempo total: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
