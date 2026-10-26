
"""Lanzador de experimentos: un comando por experimento.

Uso (desde la raíz del repo, con el entorno activo):
    python run.py                      lista los comandos
    python run.py <comando> [args]     ejecuta un comando; los args extra pasan al script

Los args extra se pasan tal cual al último script del comando, por ejemplo:
    python run.py lgd-lineal --alphas 0.1 1 10 100
    python run.py perdida-esperada --particion temporal
    python run.py pd-mlp --mode search --output-dir results/mi_busqueda --models-dir models/mi_busqueda
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DEMO_DIRS = (
    RAIZ / "results" / "demo",
    RAIZ / "models" / "demo",
)


def cmd(desc, pasos, acepta=False, limpiar=()):
    """Define un comando y sus pasos de ejecución."""
    return {
        "desc": desc,
        "pasos": [
            p if isinstance(p, tuple) else (p, [])
            for p in pasos
        ],
        "acepta": acepta,
        "limpiar": list(limpiar),
    }


COMANDOS = {
    # ---- Parte 1 (A)
    "datos": cmd(
        "Genera pd.parquet y lgd.parquet desde data/raw/loan.csv "
        "(requiere el CSV original)",
        [
            "scripts/06_generar_datos_pd.py",
            "src/lgd.py",
        ],
    ),

    "descriptivo": cmd(
        "Tablas y figuras del análisis descriptivo e hipótesis (Parte 1)",
        [
            "scripts/04_tablas_figuras_datos.py",
            "scripts/07_analisis_descriptivo.py",
        ],
    ),

    "pd-logreg": cmd(
        "[REGENERA modelos] Regresión logística de PD: búsqueda de C y modelos guardados",
        [
            "scripts/08_regresion_logistica.py",
            "scripts/09_guardar_logistica.py",
        ],
    ),

    "scorecard": cmd(
        "Scorecard de la PD (parámetros en configs/scorecard.yaml)",
        ["scripts/10_scorecard.py"],
    ),

    "tablas-pd": cmd(
        "Tabla de comparación justa de PD (logística vs. red) "
        "y tabla final de hipótesis",
        [
            (
                "scripts/11_tabla_comparacion_pd.py",
                [
                    "--mlp",
                    "results/pd_mlp_final_particionado",
                    "--salida",
                    "results/pd_comparacion_mlp_final_v3",
                    "--figuras",
                    "figures/pd_mlp_final_comparison_v3",
                ],
            ),
            "scripts/12_tabla_hipotesis_final.py",
        ],
    ),

    # ---- Redes neuronales (B)
    "pd-mlp": cmd(
        "[Persona B] MLP de PD: --mode demo|search|final y rutas de salida nuevas",
        ["scripts/14_entrenar_mlp_pd.py"],
        acepta=True,
    ),

    "pd-mlp-demo": cmd(
        "MLP de PD en modo demostración (2 épocas; escribe en results/demo y models/demo)",
        [
            (
                "scripts/14_entrenar_mlp_pd.py",
                [
                    "--mode", "demo",
                    "--output-dir", "results/demo/pd_mlp",
                    "--models-dir", "models/demo/pd_mlp",
                ],
            ),
        ],
        limpiar=[
            "results/demo/pd_mlp",
            "models/demo/pd_mlp",
        ],
    ),

    "lgd-mlp-demo": cmd(
        "MLP de LGD en modo demostración (2 épocas; escribe en results/demo y models/demo)",
        [
            (
                "scripts/16_entrenar_mlp_lgd.py",
                [
                    "--output-dir", "results/demo/lgd_mlp",
                    "--models-dir", "models/demo/lgd_mlp",
                ],
            ),
        ],
        limpiar=[
            "results/demo/lgd_mlp",
            "models/demo/lgd_mlp",
        ],
    ),

    "lgd-mlp-busqueda": cmd(
        "[Persona B] Búsqueda de 10 configuraciones del MLP de LGD (usar rutas nuevas)",
        ["scripts/19_buscar_mlp_lgd.py"],
        acepta=True,
    ),

    "lgd-mlp-final": cmd(
        "[Persona B] 30 semillas por partición del MLP de LGD "
        "(--preflight-only verifica sin entrenar)",
        ["scripts/21_entrenar_mlp_lgd_final.py"],
        acepta=True,
    ),

    # ---- Parte 2 (C)
    "lgd-lineal": cmd(
        "[REGENERA modelos] LGD: Ridge de referencia "
        "(admite --alphas, --demo, --seed)",
        ["src/lgd_lineal.py"],
        acepta=True,
    ),

    "lgd-figura": cmd(
        "LGD: figura de distribución predicha vs. observada (Ridge)",
        ["scripts/13_figura_lgd_lineal.py"],
    ),

    # CORREGIDO: ahora utiliza perdida_esperada.py
    "perdida-esperada": cmd(
        "Pérdida esperada E[L] = PD·LGD·EAD "
        "(admite --particion, --conjunto)",
        ["src/perdida_esperada.py"],
        acepta=True,
    ),

    "tests": cmd(
        "Pruebas de la scorecard",
        ["tests/test_scorecard.py"],
    ),
}


# Combinaciones. Ninguna sobrescribe modelos ni resultados finales.
COMBOS = {
    "demo": (
        "Demostración (< 2 min): Ridge, MLP de PD, MLP de LGD en modo demo "
        "+ pérdida esperada + pruebas",
        [
            ("lgd-lineal", ["--demo"]),
            ("pd-mlp-demo", []),
            ("lgd-mlp-demo", []),
            ("perdida-esperada", []),
            ("tests", []),
        ],
    ),

    "todo": (
        "Regenera tablas, figuras y pérdida esperada desde los modelos guardados "
        "(no reentrena)",
        [
            ("lgd-figura", []),
            ("perdida-esperada", ["--particion", "aleatoria"]),
            ("perdida-esperada", ["--particion", "temporal"]),
            ("tablas-pd", []),
            ("tests", []),
        ],
    ),
}


def limpiar_demo(rutas):
    """Borra solo carpetas dentro de results/demo o models/demo."""
    for r in rutas:
        p = (RAIZ / r).resolve()

        if not any(d.resolve() in p.parents for d in DEMO_DIRS):
            sys.exit(
                f"Por seguridad no se borra {p}: "
                "no está dentro de results/demo ni models/demo."
            )

        if p.exists():
            shutil.rmtree(p)


def correr(script, fijos, extra):
    """Ejecuta un script de Python y comprueba su código de salida."""
    args = [*fijos, *extra]

    print(
        f"\n>>> python {script} {' '.join(args)}".rstrip(),
        flush=True,
    )

    t0 = time.perf_counter()

    r = subprocess.run(
        [sys.executable, str(RAIZ / script), *args],
        cwd=RAIZ,
    )

    print(
        f"<<< {script}: {time.perf_counter() - t0:.1f} s",
        flush=True,
    )

    if r.returncode != 0:
        sys.exit(r.returncode)


def ejecutar(nombre, extra):
    """Ejecuta los pasos asociados a un comando."""
    c = COMANDOS[nombre]

    if extra and not c["acepta"]:
        print(
            f"Aviso: '{nombre}' no admite argumentos extra; "
            f"se ignoran: {extra}"
        )
        extra = []

    limpiar_demo(c["limpiar"])

    for i, (script, fijos) in enumerate(c["pasos"]):
        # Los argumentos extra solo van al último script del comando.
        argumentos_extra = (
            extra if i == len(c["pasos"]) - 1 else []
        )

        correr(script, fijos, argumentos_extra)


def app():
    """Inicia la aplicación web local mediante Streamlit."""
    ruta = RAIZ / "app" / "app.py"

    if not ruta.exists():
        sys.exit("La aplicación todavía no existe (falta app/app.py).")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ruta),
        ],
        cwd=RAIZ,
    )


def ayuda():
    """Muestra los comandos disponibles."""
    print(__doc__)
    print("Comandos:")

    for k, c in COMANDOS.items():
        print(f"  {k:<18} {c['desc']}")

    for k, (descripcion, _) in COMBOS.items():
        print(f"  {k:<18} {descripcion}")

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
