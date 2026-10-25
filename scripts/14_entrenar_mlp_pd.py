"""Entrena el MLP PD con los datos, preprocesador y splits compartidos del proyecto.

Modos:
  demo     Una configuración pequeña, filas completas, dos épocas como máximo.
  search   Cuatro configuraciones; selección solo por log-loss de validación.
  final    30 semillas de la configuración escogida; requiere artefactos coherentes,
           hash idéntico al search y confirmación explícita del frente de datos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
sys.path.insert(0, str(RAIZ))
from src.metricas import curva_calibracion, metricas_pd  # noqa: E402
from src.mlp_pd import MLPBinario, entrenar_una_corrida, numero_parametros, probabilidades_clase_1  # noqa: E402

PARTICIONES = {"aleatoria": "split_random", "temporal": "split_temporal"}
CONJUNTOS_EVAL = ("validacion", "prueba")
ARCHIVOS_BASE = (
    "data/processed/pd.parquet",
    "models/preprocessor.joblib",
    "models/pd_logreg_aleatoria.joblib",
    "models/pd_logreg_temporal.joblib",
    "results/regresion_logistica_resumen.json",
    "results/verificacion_datos_procesados.json",
)


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


def cargar_datos() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Any, list[str], dict[str, Any]]:
    data_path = RAIZ / "data" / "processed" / "pd.parquet"
    verification_path = RAIZ / "results" / "verificacion_datos_procesados.json"
    df = pd.read_parquet(data_path)
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
    # Una sola transformación compartida; la logística usa el mismo pre y las mismas máscaras.
    x = np.asarray(pre.transform(df[input_columns]), dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != len(df) or not np.isfinite(x).all():
        raise ValueError("La transformación del preprocesador produjo X inválido.")
    if x.shape[1] != 99:
        raise ValueError(f"Se esperaban 99 características transformadas; llegaron {x.shape[1]}.")

    for part_column in PARTICIONES.values():
        if part_column not in df:
            raise ValueError(f"El parquet no contiene {part_column}.")
        unexpected = set(df[part_column].unique()).difference({"entrenamiento", "validacion", "prueba"})
        if unexpected:
            raise ValueError(f"Valores de partición no admitidos en {part_column}: {unexpected}")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    return df, x, y, pre, input_columns, verification


def filas_particion(df: pd.DataFrame, x: np.ndarray, y: np.ndarray, column: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: (x[(mask := (df[column].to_numpy() == name))], y[mask])
            for name in ("entrenamiento", "validacion", "prueba")}


def evaluar(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, *, seed: int,
            partition: str, split_name: str, batch_size: int) -> tuple[dict[str, Any], pd.DataFrame]:
    # La sigmoid corresponde a P(y=1), que en el objetivo existente significa incumplimiento.
    p = probabilidades_clase_1(model, x, batch_size=batch_size)
    metrics = metricas_pd(y, p)
    metrics_row = {"semilla": seed, "particion": partition, "conjunto": split_name, **metrics}
    calibration = curva_calibracion(y, p, n_grupos=10)
    calibration.insert(0, "semilla", seed)
    calibration.insert(0, "conjunto", split_name)
    calibration.insert(0, "particion", partition)
    return metrics_row, calibration


def guardar_checkpoint(model: torch.nn.Module, model_dir: Path, name: str, *,
                       config: dict[str, Any], seed: int, input_dim: int,
                       input_columns: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / name
    if target.exists():
        raise FileExistsError(f"No sobrescribo modelo existente: {target}")
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": config,
        "input_dim": input_dim,
        "input_columns": input_columns,
        "seed": seed,
        "probability_of": 1,
        "target_meaning": "incumplimiento",
    }, target)


def comprobar_directorio_nuevo(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"La carpeta de salida ya tiene contenido; no sobrescribo: {path}")


def guardar_salidas(out_dir: Path, metric_rows: list[dict[str, Any]],
                    calibration_rows: list[pd.DataFrame], search_rows: list[dict[str, Any]],
                    summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_dir / "pd_mlp_metricas_por_semilla.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(out_dir / "curva_calibracion_mlp.csv", index=False)
    pd.DataFrame(search_rows).to_csv(out_dir / "pd_mlp_busqueda.csv", index=False)
    (out_dir / "pd_mlp_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def seleccionar_config(configs: list[dict[str, Any]], x_by_partition: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
                       base_config: dict[str, Any], input_dim: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], float]:
    per_config: dict[str, dict[str, Any]] = {}
    records = []
    total_seconds = 0.0
    for candidate in configs:
        per_config[candidate["id"]] = {}
        for partition, sets in x_by_partition.items():
            run = entrenar_una_corrida(
                *sets["entrenamiento"], *sets["validacion"], candidate,
                seed=int(base_config["search_seed"]), batch_size=int(base_config["batch_size"]),
                max_epochs=int(base_config["max_epochs"]), patience=int(base_config["patience"]),
                min_delta=float(base_config["min_delta"]), num_threads=int(base_config["num_threads"]))
            total_seconds += run["training_seconds"]
            val_row, _ = evaluar(run["model"], *sets["validacion"], seed=run["seed"],
                                 partition=partition, split_name="validacion",
                                 batch_size=int(base_config["batch_size"]))
            per_config[candidate["id"]][partition] = {
                "validation_logloss": float(val_row["logloss"]),
                "run": run,
            }
            records.append({"configuracion": candidate["id"], "particion": partition,
                            "semilla_busqueda": int(base_config["search_seed"]),
                            "logloss_validacion": float(val_row["logloss"]),
                            "auc_validacion": float(val_row["auc"]),
                            "brier_validacion": float(val_row["brier"]),
                            "epocas": run["epochs_run"], "epoca_mejor": run["best_epoch"],
                            "parametros_entrenables": run["parameters"],
                            "tiempo_entrenamiento_s": run["training_seconds"]})

    for candidate in configs:
        losses = [per_config[candidate["id"]][p]["validation_logloss"] for p in PARTICIONES]
        per_config[candidate["id"]]["mean_validation_logloss"] = float(np.mean(losses))
    best = min(configs, key=lambda c: per_config[c["id"]]["mean_validation_logloss"])
    return best, per_config[best["id"]], records, total_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "search", "final"), default="demo")
    parser.add_argument("--config", default="configs/mlp_pd.yaml")
    parser.add_argument("--output-dir", help="Carpeta de resultados nueva; no se sobrescribe.")
    parser.add_argument("--models-dir", help="Carpeta nueva de modelos; no se sobrescribe.")
    parser.add_argument("--search-dir", default="results/pd_mlp_provisional",
                        help="Búsqueda previa requerida por --mode final.")
    parser.add_argument("--confirmar-frente-datos", action="store_true",
                        help="Confirma que A regeneró datos/preprocesador y la logística con el protocolo sin fuga.")
    args = parser.parse_args()

    config_path = ruta_proyecto(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = hashes_entrada(config_path)
    df, x, y, pre, input_columns, verification = cargar_datos()
    fit_basis = verification.get("preprocesador_ajustado_con", "desconocido")

    if config.get("class_weight") != "none":
        raise ValueError("Este protocolo inicial es explícitamente sin balanceo; el experimento ponderado debe ser separado.")
    torch.set_num_threads(int(config["num_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if args.mode == "demo":
        output_dir = ruta_proyecto(args.output_dir or "results/pd_mlp_demo")
        model_dir = ruta_proyecto(args.models_dir or "models/pd_mlp_demo")
        demo = config["demo"]
        run_config = {**config, "batch_size": demo["batch_size"], "max_epochs": demo["max_epochs"],
                      "patience": demo["patience"], "min_delta": demo["min_delta"]}
        chosen_config = demo["config"]
        seeds = [int(demo["seed"])]
        mode_runs = "fixed_demo"
    elif args.mode == "search":
        output_dir = ruta_proyecto(args.output_dir or "results/pd_mlp_provisional")
        model_dir = ruta_proyecto(args.models_dir or "models/pd_mlp_provisional")
        run_config = config
        chosen_config = None
        seeds = [int(config["search_seed"])]
        mode_runs = "search"
    else:
        if fit_basis != "interseccion":
            raise SystemExit("Final bloqueado: el preprocesador no está ajustado solo con entrenamiento común (interseccion).")
        if not args.confirmar_frente_datos:
            raise SystemExit("Final bloqueado: coordina la regeneración del preprocesador y la logística con A y confirma con --confirmar-frente-datos.")
        search_dir = ruta_proyecto(args.search_dir)
        search_summary_path = search_dir / "pd_mlp_resumen.json"
        if not search_summary_path.is_file():
            raise FileNotFoundError(f"No existe la búsqueda requerida: {search_summary_path}")
        search_summary = json.loads(search_summary_path.read_text(encoding="utf-8"))
        if (search_summary.get("modo") != "search" or
                search_summary.get("preprocesador_ajustado_con") != "interseccion"):
            raise SystemExit("Final bloqueado: vuelve a ejecutar --mode search con el preprocesador corregido.")
        if search_summary.get("seleccion_final_habilitada") is not True:
            raise SystemExit("Final bloqueado: la búsqueda global actual puede solapar validaciones y pruebas entre particiones; corrige el protocolo de selección antes de usar su arquitectura.")
        if search_summary.get("hashes_entrada") != hashes:
            raise SystemExit("Final bloqueado: los datos, preprocesador, logística o configuración cambiaron después de la búsqueda; repítela.")
        output_dir = ruta_proyecto(args.output_dir or "results/pd_mlp_final")
        model_dir = ruta_proyecto(args.models_dir or "models/pd_mlp_final")
        run_config = config
        chosen_config = search_summary["configuracion_elegida"]
        seeds = [int(seed) for seed in config["final_seeds"]]
        mode_runs = "final_seeds"

    comprobar_directorio_nuevo(output_dir)
    comprobar_directorio_nuevo(model_dir)

    x_by_partition = {
        label: filas_particion(df, x, y, column)
        for label, column in PARTICIONES.items()
    }
    rows_by_partition = {
        label: {name: int(len(pair[1])) for name, pair in sets.items()}
        for label, sets in x_by_partition.items()
    }

    wall_start = time.perf_counter()
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    search_rows: list[dict[str, Any]] = []
    fit_seconds = 0.0
    selected_config = chosen_config
    config_count = 1

    if mode_runs == "search":
        config_count = len(config["search"])
        selected_config, best_by_partition, search_rows, fit_seconds = seleccionar_config(
            config["search"], x_by_partition, config, x.shape[1])
        configs_to_train = [(seed, partition, best_by_partition[partition]["run"])
                            for seed in seeds for partition in PARTITIONES_ORDER]
    elif mode_runs == "fixed_demo":
        selected_config = chosen_config
        demo_runs = {}
        for partition, sets in x_by_partition.items():
            run = entrenar_una_corrida(
                *sets["entrenamiento"], *sets["validacion"], selected_config,
                seed=seeds[0], batch_size=int(run_config["batch_size"]),
                max_epochs=int(run_config["max_epochs"]), patience=int(run_config["patience"]),
                min_delta=float(run_config["min_delta"]), num_threads=int(config["num_threads"]))
            fit_seconds += run["training_seconds"]
            demo_runs[partition] = run
            search_rows.append({"configuracion": selected_config["id"], "particion": partition,
                                "semilla_busqueda": seeds[0],
                                "logloss_validacion": run["best_validation_logloss"],
                                "epocas": run["epochs_run"], "epoca_mejor": run["best_epoch"],
                                "parametros_entrenables": run["parameters"],
                                "tiempo_entrenamiento_s": run["training_seconds"]})
        configs_to_train = [(seeds[0], partition, demo_runs[partition])
                            for partition in PARTITIONES_ORDER]
    else:
        search_summary = json.loads((ruta_proyecto(args.search_dir) / "pd_mlp_resumen.json").read_text(encoding="utf-8"))
        config_count = int(search_summary["configuraciones_entrenadas"])
        fit_seconds = 0.0
        configs_to_train = []
        for seed in seeds:
            for partition, sets in x_by_partition.items():
                run = entrenar_una_corrida(
                    *sets["entrenamiento"], *sets["validacion"], selected_config,
                    seed=seed, batch_size=int(config["batch_size"]), max_epochs=int(config["max_epochs"]),
                    patience=int(config["patience"]), min_delta=float(config["min_delta"]),
                    num_threads=int(config["num_threads"]))
                fit_seconds += run["training_seconds"]
                configs_to_train.append((seed, partition, run))

    # Search guarda el mejor checkpoint seleccionado por validación; demo conserva su única configuración.
    if mode_runs in ("search", "fixed_demo"):
        if mode_runs == "search":
            for partition in PARTITIONES_ORDER:
                run = best_by_partition[partition]["run"]
                guardar_checkpoint(run["model"], model_dir, f"pd_mlp_{partition}_search_seed_{seeds[0]}.pt",
                                   config=selected_config, seed=seeds[0], input_dim=x.shape[1], input_columns=input_columns)
        else:
            for seed, partition, run in configs_to_train:
                guardar_checkpoint(run["model"], model_dir, f"pd_mlp_{partition}_demo_seed_{seeds[0]}.pt",
                                   config=selected_config, seed=seeds[0], input_dim=x.shape[1], input_columns=input_columns)
        # La demo tiene una corrida por partición y conserva todas las filas de cada split.
    else:
        for seed, partition, run in configs_to_train:
            guardar_checkpoint(run["model"], model_dir, f"pd_mlp_{partition}_seed_{seed}.pt",
                               config=selected_config, seed=seed, input_dim=x.shape[1], input_columns=input_columns)

    # La prueba se calcula después de fijar la configuración, nunca interviene en selección.
    for seed, partition, run in configs_to_train or []:
        sets = x_by_partition[partition]
        for split_name in CONJUNTOS_EVAL:
            row, calibration = evaluar(run["model"], *sets[split_name], seed=seed,
                                       partition=partition, split_name=split_name,
                                       batch_size=int(run_config["batch_size"]))
            metric_rows.append(row)
            calibration_rows.append(calibration)

    elapsed = time.perf_counter() - wall_start
    selected_losses = {p: float(best_by_partition[p]["validation_logloss"])
                       for p in PARTITIONES_ORDER} if mode_runs == "search" else None
    mean_selected_loss = (float(np.mean(list(selected_losses.values())))
                          if selected_losses is not None else None)
    provisional = args.mode != "final" or fit_basis != "interseccion"
    search_seconds = fit_seconds if mode_runs != "final_seeds" else float(search_summary["tiempo_total_busqueda_s"])
    if mode_runs == "fixed_demo":
        # No hubo búsqueda; reporta el tiempo de las dos corridas demo.
        search_seconds = fit_seconds
    summary = {
        "modo": args.mode,
        "provisional": provisional,
        "seleccion_final_habilitada": False,
        "estado_protocolo_seleccion": "provisional_solapamiento_entre_particiones_aleatoria_y_temporal",
        "motivo_provisional": ("Demo/búsqueda inicial; preprocesador actual ajustado con entrenamiento aleatorio, "
                               "que incluye filas de validación/prueba temporal. La selección global también puede "
                               "solapar validaciones y pruebas entre particiones." if provisional else None),
        "preprocesador_ajustado_con": fit_basis,
        "filas_por_particion": rows_by_partition,
        "columnas_originales_preprocesador": input_columns,
        "columnas_transformadas": int(x.shape[1]),
        "probabilidad_reportada": "sigmoid(logit) = P(objetivo=1 = incumplimiento)",
        "perdida_entrenamiento": "BCEWithLogitsLoss (verosimilitud Bernoulli)",
        "criterio_parada_temprana": "log-loss no ponderado en validación",
        "criterio_seleccion": "promedio simple de log-loss de validación aleatoria y temporal; sin consultar prueba",
        "class_weight": "none",
        "configuracion_elegida": selected_config,
        "logloss_validacion_por_particion": selected_losses,
        "logloss_validacion_promedio": mean_selected_loss,
        "configuraciones_entrenadas": config_count,
        "tiempo_total_busqueda_s": round(search_seconds, 6),
        "tiempo_total_entrenamiento_final_s": round(fit_seconds, 6) if mode_runs == "final_seeds" else None,
        "tiempo_total_ejecucion_s": round(elapsed, 6),
        "parametros_entrenables": numero_parametros(MLPBinario(x.shape[1], selected_config)),
        "semillas": seeds,
        "dispositivo": "cpu",
        "versiones": {"python": sys.version.split()[0], "torch": torch.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__},
        "hashes_entrada": hashes,
    }
    guardar_salidas(output_dir, metric_rows, calibration_rows, search_rows, summary)
    print(json.dumps({"modo": args.mode, "salida": str(output_dir), "modelos": str(model_dir),
                      "provisional": provisional, "configuracion": selected_config["id"],
                      "filas_por_particion": rows_by_partition,
                      "tiempo_total_ejecucion_s": round(elapsed, 3),
                      "tiempo_total_entrenamiento_s": round(fit_seconds, 3),
                      "parametros_entrenables": summary["parametros_entrenables"],
                      "metricas": metric_rows}, indent=2, ensure_ascii=False))
    return 0


PARTITIONES_ORDER = ("aleatoria", "temporal")
if __name__ == "__main__":
    raise SystemExit(main())
