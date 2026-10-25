"""Búsqueda y entrenamiento del MLP de PD sobre los splits compartidos.

Modos: demo (integración), search (validación exploratoria por partición) y
final (30+ semillas por partición con las configuraciones ya fijadas).
La búsqueda nunca usa prueba y no mezcla validaciones entre protocolos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

RAIZ = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(RAIZ / ".venv" / "matplotlib-config"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(RAIZ))
from src.metricas import curva_calibracion, metricas_pd  # noqa: E402
from src.mlp_pd import MLPBinario, entrenar_una_corrida, numero_parametros, probabilidades_clase_1  # noqa: E402

PARTICIONES = {"aleatoria": "split_random", "temporal": "split_temporal"}
CONJUNTOS = ("entrenamiento", "validacion", "prueba")
CONJUNTOS_EVAL = ("validacion", "prueba")
ARCHIVOS_BASE = (
    "data/processed/pd.parquet",
    "models/preprocessor.joblib",
    "models/pd_logreg_aleatoria.joblib",
    "models/pd_logreg_temporal.joblib",
    "results/regresion_logistica_resumen.json",
    "results/verificacion_datos_procesados.json",
    "scripts/14_entrenar_mlp_pd.py",
    "src/mlp_pd.py",
    "src/metricas.py",
)
COLUMNAS_METRICAS = ["semilla", "particion", "conjunto", "auc", "gini", "ks", "brier", "logloss"]


def ruta_proyecto(valor: str | Path) -> Path:
    p = Path(valor)
    return p if p.is_absolute() else RAIZ / p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes_entrada(config_path: Path) -> dict[str, str]:
    paths = [RAIZ / p for p in ARCHIVOS_BASE] + [config_path]
    missing = [str(p.relative_to(RAIZ)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Faltan artefactos de entrada requeridos: " + ", ".join(missing))
    return {str(p.relative_to(RAIZ)): sha256(p) for p in paths}


def cargar_datos() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    df = pd.read_parquet(RAIZ / "data" / "processed" / "pd.parquet")
    y = df["objetivo"].to_numpy(dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("El parquet debe tener ambas clases, codificadas como 0 y 1.")
    pre = joblib.load(RAIZ / "models" / "preprocessor.joblib")
    input_columns = list(pre.feature_names_in_)
    forbidden = {"objetivo", *PARTICIONES.values()}
    leaked = sorted(forbidden.intersection(input_columns))
    if leaked:
        raise ValueError(f"Objetivo o máscaras de partición aparecen como predictores: {leaked}")
    absent = sorted(set(input_columns).difference(df.columns))
    if absent:
        raise ValueError(f"Faltan columnas requeridas por el preprocesador: {absent}")
    # El mismo artefacto y las mismas columnas del baseline logístico; transformación única.
    x = np.asarray(pre.transform(df[input_columns]), dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != len(df) or not np.isfinite(x).all():
        raise ValueError("La transformación produjo una matriz inválida.")
    if x.shape[1] != 99:
        raise ValueError(f"Se esperaban 99 columnas transformadas; llegaron {x.shape[1]}.")
    for col in PARTICIONES.values():
        if col not in df:
            raise ValueError(f"El parquet no contiene {col}.")
        if set(df[col].unique()) != set(CONJUNTOS):
            raise ValueError(f"Valores de partición inesperados en {col}.")
    verification = json.loads((RAIZ / "results" / "verificacion_datos_procesados.json").read_text(encoding="utf-8"))
    return df, x, y, input_columns, verification


def filas_particion(df: pd.DataFrame, x: np.ndarray, y: np.ndarray, column: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    labels = df[column].to_numpy()
    return {name: (x[mask := labels == name], y[mask]) for name in CONJUNTOS}


def auditar_cruces(df: pd.DataFrame) -> dict[str, int]:
    random_split, temporal_split = df["split_random"], df["split_temporal"]
    return {
        "validacion_aleatoria_en_prueba_temporal": int(((random_split == "validacion") & (temporal_split == "prueba")).sum()),
        "prueba_aleatoria_en_validacion_temporal": int(((random_split == "prueba") & (temporal_split == "validacion")).sum()),
    }


def comprobar_seleccion_por_particion(df: pd.DataFrame) -> None:
    """Falla si cada protocolo no tiene sus propios train/val/test disjuntos."""
    for partition, column in PARTICIONES.items():
        masks = {name: (df[column].to_numpy() == name) for name in CONJUNTOS}
        for left, right in (("entrenamiento", "validacion"), ("entrenamiento", "prueba"), ("validacion", "prueba")):
            if np.any(masks[left] & masks[right]):
                raise ValueError(f"Las máscaras {left}/{right} se cruzan en la partición {partition}.")
    # El cruce ENTRE protocolos puede existir; la selección se hará dentro de cada bucle de partición.


def evaluar(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, *, seed: int,
            partition: str, split_name: str, batch_size: int) -> tuple[dict[str, Any], pd.DataFrame]:
    p = probabilidades_clase_1(model, x, batch_size=batch_size)
    row = {"semilla": seed, "particion": partition, "conjunto": split_name, **metricas_pd(y, p)}
    calibration = curva_calibracion(y, p, n_grupos=10)
    calibration.insert(0, "semilla", seed)
    calibration.insert(0, "conjunto", split_name)
    calibration.insert(0, "particion", partition)
    return row, calibration


def guardar_checkpoint(model: torch.nn.Module, model_dir: Path, name: str, *,
                       config: dict[str, Any], seed: int, input_dim: int,
                       input_columns: list[str], hashes: dict[str, str],
                       partition: str, mode: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / name
    if target.exists():
        raise FileExistsError(f"No sobrescribo modelo existente: {target}")
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": config, "input_dim": input_dim, "input_columns": input_columns,
        "seed": seed, "partition": partition, "mode": mode, "hashes_entrada": hashes,
        "probability_of": 1, "target_meaning": "incumplimiento",
    }, target)


def comprobar_directorio_nuevo(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"La carpeta ya tiene contenido; no sobrescribo: {path}")


def versiones() -> dict[str, str]:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def seleccionar_config(configs: list[dict[str, Any]], seeds: list[int],
                       x_by_partition: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
                       base_config: dict[str, Any], input_dim: int,
                       input_columns: list[str], hashes: dict[str, str], model_dir: Path
                       ) -> tuple[dict[str, dict[str, Any]], dict[str, float], list[dict[str, Any]], list[pd.DataFrame], float]:
    """Escoge por menor log-loss media de semillas exploratorias, sin agrupar particiones."""
    selected: dict[str, dict[str, Any]] = {}
    times: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    total_seconds = 0.0
    for partition, sets in x_by_partition.items():
        if set(sets) != set(CONJUNTOS):
            raise AssertionError(f"Conjuntos inesperados en {partition}.")
        per_candidate: dict[str, list[dict[str, Any]]] = {c["id"]: [] for c in configs}
        partition_seconds = 0.0
        for candidate in configs:
            for seed in seeds:
                run = entrenar_una_corrida(
                    *sets["entrenamiento"], *sets["validacion"], candidate,
                    seed=seed, batch_size=int(base_config["batch_size"]),
                    max_epochs=int(base_config["max_epochs"]), patience=int(base_config["patience"]),
                    min_delta=float(base_config["min_delta"]), num_threads=int(base_config["num_threads"]))
                dt = float(run["training_seconds"])
                total_seconds += dt
                partition_seconds += dt
                val_row, cal = evaluar(run["model"], *sets["validacion"], seed=seed,
                                       partition=partition, split_name="validacion",
                                       batch_size=int(base_config["batch_size"]))
                if val_row["particion"] != partition or val_row["conjunto"] != "validacion":
                    raise AssertionError("La selección solo admite validación de la partición que se está evaluando.")
                cal.insert(3, "configuracion", candidate["id"])
                calibration_rows.append(cal)
                item = {**val_row, "configuracion": candidate["id"],
                        "configuracion_json": json.dumps(candidate, ensure_ascii=False, sort_keys=True),
                        "hipotesis": candidate.get("hypothesis", ""),
                        "cambio_frente_base": candidate.get("changed_from_base", ""),
                        "semilla_exploratoria": seed, "epoca_mejor": run["best_epoch"],
                        "epocas_ejecutadas": run["epochs_run"],
                        "parametros_entrenables": run["parameters"],
                        "tiempo_entrenamiento_s": dt,
                        "versiones_json": json.dumps(versiones(), sort_keys=True),
                        "hashes_entrada_json": json.dumps(hashes, sort_keys=True)}
                per_candidate[candidate["id"]].append(item)
                records.append(item)
                guardar_checkpoint(run["model"], model_dir,
                                   f"{partition}__{candidate['id']}__seed_{seed}.pt",
                                   config=candidate, seed=seed, input_dim=input_dim,
                                   input_columns=input_columns,
                                   hashes=hashes, partition=partition, mode="search")
        for candidate in configs:
            candidate_records = per_candidate[candidate["id"]]
            candidate["_mean_val_logloss"] = float(np.mean([r["logloss"] for r in candidate_records]))
        # Assertion documents and protects that exactly one partition's validations form each ranking.
        ranked = sorted(configs, key=lambda c: c["_mean_val_logloss"])
        selected[partition] = {k: v for k, v in ranked[0].items() if not k.startswith("_")}
        for candidate in configs:
            for row in per_candidate[candidate["id"]]:
                row["logloss_media_config_particion"] = candidate["_mean_val_logloss"]
                row["seleccionada_en_particion"] = candidate["id"] == ranked[0]["id"]
        times[partition] = partition_seconds
        # Delete temporary scores before next partition; never combine validation sets.
        for candidate in configs:
            candidate.pop("_mean_val_logloss", None)
    return selected, times, records, calibration_rows, total_seconds


def figura_busqueda(records: list[dict[str, Any]], selected: dict[str, dict[str, Any]], output: Path) -> None:
    tab = pd.DataFrame(records)
    fig, axes = plt.subplots(1, len(PARTICIONES), figsize=(max(12, 5 * len(PARTICIONES)), 5), squeeze=False)
    for ax, partition in zip(axes[0], PARTICIONES):
        p = tab[tab["particion"] == partition].groupby("configuracion").agg(
            logloss=("logloss", "mean"), sd=("logloss", "std"), auc=("auc", "mean"))
        p = p.sort_values("logloss")
        colors = ["#0072B2" if k == selected[partition]["id"] else "#999999" for k in p.index]
        ax.bar(range(len(p)), p["logloss"], yerr=p["sd"].fillna(0), color=colors, capsize=2)
        ax.set_xticks(range(len(p)), p.index, rotation=65, ha="right")
        ax.set_ylabel("Log-loss media de validación (menor es mejor)")
        ax.set_title(f"{partition}: elegida {selected[partition]['id']}")
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def figura_calibracion(calibration: list[pd.DataFrame], output: Path) -> None:
    if not calibration:
        return
    c = pd.concat(calibration, ignore_index=True)
    c = c[c["conjunto"] == "prueba"]
    if c.empty:
        return
    fig, axes = plt.subplots(1, len(PARTICIONES), figsize=(10, 4.5), squeeze=False)
    for ax, partition in zip(axes[0], PARTICIONES):
        p = c[c["particion"] == partition].groupby("grupo").agg(
            pred=("prob_predicha_media", "mean"), obs=("tasa_observada", "mean"),
            obs_min=("tasa_observada", "min"), obs_max=("tasa_observada", "max"))
        hi = float(max(p["pred"].max(), p["obs_max"].max())) * 1.05
        ax.plot([0, hi], [0, hi], "--", color="gray", label="Calibración ideal")
        ax.plot(p["pred"], p["obs"], "o-", label="MLP (media semilla)" if c["semilla"].nunique() > 1 else "MLP")
        if c["semilla"].nunique() > 1:
            ax.fill_between(p["pred"], p["obs_min"], p["obs_max"], alpha=.2)
        ax.set(xlabel="PD media predicha por decil", ylabel="Incumplimiento observado", title=partition,
               xlim=(0, hi), ylim=(0, hi))
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def guardar_salidas(output: Path, *, mode: str, metric_rows: list[dict[str, Any]],
                    calibration: list[pd.DataFrame], search_rows: list[dict[str, Any]],
                    summary: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if mode != "search":
        pd.DataFrame(metric_rows, columns=COLUMNAS_METRICAS).to_csv(output / "pd_mlp_metricas_por_semilla.csv", index=False)
        pd.concat(calibration, ignore_index=True).to_csv(output / "curva_calibracion_mlp.csv", index=False)
        figura_calibracion(calibration, output / "figura_calibracion_mlp.png")
    if mode == "search":
        pd.DataFrame(search_rows).to_csv(output / "pd_mlp_busqueda.csv", index=False)
        pd.concat(calibration, ignore_index=True).to_csv(output / "curva_calibracion_busqueda.csv", index=False)
    elif search_rows:
        pd.DataFrame(search_rows).to_csv(output / "pd_mlp_busqueda.csv", index=False)
    (output / "pd_mlp_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if mode == "search":
        figura_busqueda(search_rows, summary["configuraciones_elegidas"], output / "figura_busqueda_pd_mlp.png")


def preprocesador_estado(verification: dict[str, Any]) -> dict[str, Any]:
    basis = verification.get("preprocesador_ajustado_con", "desconocido")
    temporal_leak = basis == "entrenamiento_aleatorio"
    description = {
        "entrenamiento_aleatorio": "Ajustado con split_random=entrenamiento; incluye registros que son validación/prueba temporal.",
        "interseccion": "Ajustado con entrenamiento aleatorio ∩ entrenamiento temporal.",
    }.get(basis, "Base de ajuste no identificada en los metadatos disponibles.")
    return {"artefacto": "models/preprocessor.joblib", "base_de_ajuste_registrada": basis,
            "descripcion": description, "limitacion_evaluacion_temporal": temporal_leak,
            "decision_equipo": "Se conserva el artefacto actual por decisión comunicada del equipo; no equivale a demostrar ausencia de fuga.",
            "evidencia_prueba_alternativa_a": "No se encontró en el repositorio informe, métricas ni definición exacta de la alternativa probada.",
            "metricas_y_diferencia_alternativa": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "search", "final"), default="demo")
    parser.add_argument("--config", default="configs/mlp_pd.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--models-dir")
    parser.add_argument("--search-dir", default="results/pd_mlp_search_particionado")
    args = parser.parse_args()
    config_path = ruta_proyecto(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = hashes_entrada(config_path)
    df, x, y, input_columns, verification = cargar_datos()
    comprobar_seleccion_por_particion(df)
    overlaps = auditar_cruces(df)
    pre_state = preprocesador_estado(verification)
    fit_basis = pre_state["base_de_ajuste_registrada"]
    if config.get("class_weight") != "none":
        raise ValueError("La búsqueda registrada es sin balanceo; ponderación debe ser un experimento separado.")
    if len(config["search_seeds"]) < 2 or len(set(config["search_seeds"])) != len(config["search_seeds"]):
        raise ValueError("La búsqueda requiere al menos dos semillas exploratorias únicas.")
    torch.set_num_threads(int(config["num_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    search_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    selected_configs: dict[str, dict[str, Any]]
    search_seconds_by_partition: dict[str, float] = {p: 0.0 for p in PARTICIONES}
    search_seconds_total = 0.0
    final_training_seconds = 0.0
    config_count = 1
    if args.mode == "demo":
        output_dir = ruta_proyecto(args.output_dir or "results/pd_mlp_demo")
        model_dir = ruta_proyecto(args.models_dir or "models/pd_mlp_demo")
        demo = config["demo"]
        run_params = {**config, "batch_size": demo["batch_size"], "max_epochs": demo["max_epochs"],
                      "patience": demo["patience"], "min_delta": demo["min_delta"]}
        selected_configs = {p: demo["config"] for p in PARTICIONES}
        seeds = [int(demo["seed"])]
    elif args.mode == "search":
        output_dir = ruta_proyecto(args.output_dir or "results/pd_mlp_search_particionado")
        model_dir = ruta_proyecto(args.models_dir or "models/pd_mlp_search_particionado")
        run_params = config
        seeds = [int(s) for s in config["search_seeds"]]
        config_count = len(config["search"])
    else:
        search_dir = ruta_proyecto(args.search_dir)
        search_summary_path = search_dir / "pd_mlp_resumen.json"
        if not search_summary_path.is_file():
            raise FileNotFoundError(f"No existe búsqueda: {search_summary_path}")
        search_summary = json.loads(search_summary_path.read_text(encoding="utf-8"))
        if search_summary.get("modo") != "search" or search_summary.get("protocolo_seleccion") != "independiente_por_particion":
            raise SystemExit("Final bloqueado: la búsqueda no acredita selección independiente por partición.")
        if search_summary.get("seleccion_final_habilitada") is not True:
            raise SystemExit("Final bloqueado: el manifiesto no habilita las configuraciones seleccionadas.")
        if search_summary.get("hashes_entrada") != hashes:
            raise SystemExit("Final bloqueado: cambió algún hash de datos, preprocesador, logística o configuración; repite la búsqueda.")
        if search_summary.get("preprocesador", {}).get("base_de_ajuste_registrada") != fit_basis:
            raise SystemExit("Final bloqueado: cambió el registro del preprocesador desde la búsqueda.")
        selected_configs = search_summary.get("configuraciones_elegidas", {})
        if set(selected_configs) != set(PARTICIONES):
            raise SystemExit("Final bloqueado: falta configuración elegida para cada partición.")
        output_dir = ruta_proyecto(args.output_dir or "results/pd_mlp_final_particionado")
        model_dir = ruta_proyecto(args.models_dir or "models/pd_mlp_final_particionado")
        run_params = config
        seeds = [int(s) for s in config["final_seeds"]]
        if len(seeds) < 30 or len(set(seeds)) != len(seeds):
            raise ValueError("El modo final necesita al menos 30 semillas únicas.")
        config_count = int(search_summary["configuraciones_entrenadas"])
        search_seconds_by_partition = search_summary["tiempo_total_busqueda_s_por_particion"]
        search_seconds_total = float(search_summary["tiempo_total_busqueda_s"])

    comprobar_directorio_nuevo(output_dir)
    comprobar_directorio_nuevo(model_dir)
    x_by_partition = {p: filas_particion(df, x, y, col) for p, col in PARTICIONES.items()}
    rows_by_partition = {p: {s: int(len(v[1])) for s, v in data.items()} for p, data in x_by_partition.items()}
    wall_start = time.perf_counter()

    if args.mode == "search":
        selected_configs, search_seconds_by_partition, search_rows, calibration_rows, search_seconds_total = seleccionar_config(
            config["search"], seeds, x_by_partition, config, x.shape[1], input_columns, hashes, model_dir)
        # El manifiesto se crea solo después de verificar que ambas selecciones existen de forma separada.
        if set(selected_configs) != set(PARTICIONES):
            raise AssertionError("La selección no produjo una configuración por partición.")
        for row in search_rows:
            partition = row["particion"]
            row["configuracion_seleccionada"] = row["configuracion"] == selected_configs[partition]["id"]
    else:
        mode_seed = seeds[0] if args.mode == "demo" else None
        partitions_to_run = list(PARTICIONES) if args.mode == "demo" else list(PARTICIONES)
        for seed in ([mode_seed] if args.mode == "demo" else seeds):
            for partition in partitions_to_run:
                sets = x_by_partition[partition]
                cfg = selected_configs[partition]
                run = entrenar_una_corrida(
                    *sets["entrenamiento"], *sets["validacion"], cfg, seed=int(seed),
                    batch_size=int(run_params["batch_size"]), max_epochs=int(run_params["max_epochs"]),
                    patience=int(run_params["patience"]), min_delta=float(run_params["min_delta"]),
                    num_threads=int(config["num_threads"]))
                final_training_seconds += run["training_seconds"]
                checkpoint_tag = "demo" if args.mode == "demo" else "final"
                guardar_checkpoint(run["model"], model_dir,
                                   f"pd_mlp_{partition}_{checkpoint_tag}_seed_{seed}.pt",
                                   config=cfg, seed=int(seed), input_dim=x.shape[1], input_columns=input_columns,
                                   hashes=hashes, partition=partition, mode=args.mode)
                eval_sets = ("validacion",) if args.mode == "demo" else CONJUNTOS_EVAL
                for split_name in eval_sets:
                    row, cal = evaluar(run["model"], *sets[split_name], seed=int(seed),
                                       partition=partition, split_name=split_name,
                                       batch_size=int(run_params["batch_size"]))
                    metric_rows.append(row)
                    calibration_rows.append(cal)
                if args.mode == "demo":
                    search_rows.append({"configuracion": cfg["id"], "particion": partition,
                                        "semilla_exploratoria": seed,
                                        "configuracion_json": json.dumps(cfg, sort_keys=True),
                                        "epoca_mejor": run["best_epoch"], "epocas_ejecutadas": run["epochs_run"],
                                        "parametros_entrenables": run["parameters"],
                                        "tiempo_entrenamiento_s": run["training_seconds"]})

    wall_elapsed = time.perf_counter() - wall_start
    params_by_partition = {p: numero_parametros(MLPBinario(x.shape[1], selected_configs[p])) for p in PARTICIONES}
    is_final = args.mode == "final"
    summary = {
        "modo": args.mode,
        "provisional": bool(pre_state["limitacion_evaluacion_temporal"] or args.mode != "final"),
        "motivo_provisional": "El preprocesador vigente se conserva por decisión comunicada del equipo; temporalmente fue ajustado con entrenamiento aleatorio, que incluye filas temporales de validación/prueba." if pre_state["limitacion_evaluacion_temporal"] else None,
        "protocolo_seleccion": "independiente_por_particion",
        "seleccion_final_habilitada": args.mode == "search",
        "criterio_seleccion": "mínima media de log-loss de validación entre semillas exploratorias, calculada por separado dentro de cada partición; no se consultan pruebas ni se agregan validaciones de protocolos distintos",
        "configuraciones_elegidas": selected_configs,
        "configuracion_elegida": selected_configs if args.mode != "demo" else None,
        "logloss_validacion_media_por_particion": {
            p: float(np.mean([r["logloss"] for r in search_rows if r.get("particion") == p and r.get("configuracion") == selected_configs[p]["id"]]))
            for p in PARTICIONES
        } if args.mode == "search" else None,
        "preprocesador": pre_state,
        "preprocesador_ajustado_con": fit_basis,
        "limitacion_temporal_preprocesador": pre_state["limitacion_evaluacion_temporal"],
        "cruces_entre_particiones_no_usados_en_seleccion": overlaps,
        "class_weight": "none",
        "filas_por_particion": rows_by_partition,
        "columnas_originales_preprocesador": input_columns,
        "columnas_transformadas": int(x.shape[1]),
        "probabilidad_reportada": "sigmoid(logit) = P(objetivo=1 = incumplimiento)",
        "perdida_entrenamiento": "BCEWithLogitsLoss",
        "criterio_parada_temprana": "log-loss de validación de la partición correspondiente",
        "configuraciones_entrenadas": config_count,
        "configuraciones_entrenadas_por_particion": {p: config_count for p in PARTICIONES},
        "semillas_exploratorias": config.get("search_seeds", []),
        "semillas_finales": seeds if is_final else [],
        "parametros_entrenables": params_by_partition,
        "tiempo_total_busqueda_s": round(search_seconds_total, 6),
        "tiempo_total_busqueda_s_por_particion": {p: round(v, 6) for p, v in search_seconds_by_partition.items()},
        "tiempo_total_entrenamiento_final_s": round(final_training_seconds, 6) if is_final else None,
        "tiempo_total_ejecucion_s": round(wall_elapsed, 6),
        "presupuesto_cpu_minutos_declarado": config["planned_cpu_budget_minutes"],
        "dispositivo": "cpu",
        "versiones": versiones(),
        "hashes_entrada": hashes,
    }
    guardar_salidas(output_dir, mode=args.mode, metric_rows=metric_rows,
                    calibration=calibration_rows, search_rows=search_rows, summary=summary)
    print(json.dumps({"modo": args.mode, "salida": str(output_dir), "modelos": str(model_dir),
                      "protocolo_seleccion": summary["protocolo_seleccion"],
                      "configuraciones_elegidas": {p: c["id"] for p, c in selected_configs.items()},
                      "provisional_temporal": summary["limitacion_temporal_preprocesador"],
                      "filas_por_particion": rows_by_partition,
                      "tiempo_ejecucion_s": round(wall_elapsed, 3),
                      "tiempo_entrenamiento_s": round(final_training_seconds, 3),
                      "metricas": metric_rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
