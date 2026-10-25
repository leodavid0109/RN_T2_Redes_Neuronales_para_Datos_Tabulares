"""Demo de MLP para LGD sobre exactamente la muestra y particiones de la Ridge.

Solo ejecuta la base configurable de demostración, evalúa validación y no lee la prueba
para elegir configuración. Produce salidas nuevas en results/lgd_mlp_demo y models/lgd_mlp_demo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv" / "matplotlib-config"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(ROOT))
from src.metricas_lgd import metricas_lgd  # noqa: E402
from src.mlp_lgd import MLPRegresorLGD, entrenar_una_corrida, numero_parametros, predecir_lgd  # noqa: E402

PARTITIONS = {"aleatoria": "split_random", "temporal": "split_temporal"}
SETS = ("entrenamiento", "validacion", "prueba")
HASH_PATHS = (
    "data/processed/pd.parquet", "data/processed/lgd.parquet", "models/preprocessor.joblib",
    "models/lgd_lineal_aleatoria.joblib", "models/lgd_lineal_temporal.joblib",
    "results/lgd_lineal.csv", "results/lgd_lineal_pred.parquet",
    "results/lgd_lineal_resumen.json", "results/verificacion_datos_procesados.json",
    "configs/mlp_lgd.yaml", "src/lgd_lineal.py", "src/metricas_lgd.py",
    "src/mlp_lgd.py", "scripts/16_entrenar_mlp_lgd.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def asegurar_salida_nueva(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"No sobrescribo resultados existentes: {path}")


def resumen_distribucion(values: np.ndarray, observed: np.ndarray, series: str, partition: str) -> dict:
    return {
        "particion": partition, "serie": series, "n": int(len(values)),
        "media": float(np.mean(values)), "desv_est": float(np.std(values, ddof=1)),
        "mediana": float(np.median(values)), "p90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "masa_exacta_0": float(np.mean(values == 0.0)),
        "masa_exacta_1": float(np.mean(values == 1.0)),
        "tasa_observada_media": float(np.mean(observed)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mlp_lgd.yaml")
    parser.add_argument("--output-dir", default="results/lgd_mlp_demo")
    parser.add_argument("--models-dir", default="models/lgd_mlp_demo")
    args = parser.parse_args()
    config_path = Path(args.config)
    out_dir, model_dir = Path(args.output_dir), Path(args.models_dir)
    if not config_path.is_absolute(): config_path = ROOT / config_path
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    if not model_dir.is_absolute(): model_dir = ROOT / model_dir
    asegurar_salida_nueva(out_dir)
    asegurar_salida_nueva(model_dir)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    demo = config["demo"]
    model_config = demo["model"]
    if config["output_activation"] != "sigmoid" or config["loss"] != "mse":
        raise ValueError("Este esqueleto inicial fija salida sigmoid y pérdida MSE")
    torch.set_num_threads(int(config["num_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data = pd.read_parquet(ROOT / "data/processed/lgd.parquet")
    data = data.loc[data["castigado"].astype(bool)].copy().reset_index(drop=True)
    if data["id"].duplicated().any():
        raise ValueError("La muestra castigada contiene id duplicados")
    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
    input_columns = list(preprocessor.feature_names_in_)
    missing = sorted(set(input_columns).difference(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas de entrada del preprocesador: {missing}")
    # Es el mismo transform y muestra que usa src/lgd_lineal.py tras su filtro castigado.
    x = np.asarray(preprocessor.transform(data), dtype=np.float32)
    y = data["lgd"].to_numpy(dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != int(config["input_features"]) or not np.isfinite(x).all():
        raise ValueError(f"Se esperaba una matriz finita de {config['input_features']} columnas")
    if not np.isfinite(y).all() or np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("LGD debe ser finita y pertenecer a [0,1]")

    ridge_predictions = pd.read_parquet(ROOT / "results/lgd_lineal_pred.parquet")
    ridge_metrics = pd.read_csv(ROOT / "results/lgd_lineal.csv")
    hashes = {p: sha256(ROOT / p) for p in HASH_PATHS}
    all_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    distribution_rows: list[dict] = []
    validation_history: dict[str, list[dict]] = {}
    total_train_seconds = 0.0
    wall_start = time.perf_counter()

    for partition, split_column in PARTITIONS.items():
        labels = data[split_column].to_numpy()
        indices = {split: np.flatnonzero(labels == split) for split in SETS}
        # Verifica por IDs que cada grupo coincide exactamente con la Ridge guardada.
        for split in SETS:
            ridge_ids = set(ridge_predictions.loc[
                (ridge_predictions["particion"] == partition)
                & (ridge_predictions["conjunto"] == split), "id"].tolist())
            mlp_ids = set(data.iloc[indices[split]]["id"].tolist())
            if ridge_ids != mlp_ids:
                raise ValueError(f"Los ids {partition}/{split} difieren de Ridge: {len(mlp_ids)} vs {len(ridge_ids)}")

        train_i, valid_i = indices["entrenamiento"], indices["validacion"]
        run = entrenar_una_corrida(
            x[train_i], y[train_i], x[valid_i], y[valid_i], model_config,
            seed=int(demo["seed"]), batch_size=int(demo["batch_size"]),
            max_epochs=int(demo["max_epochs"]), patience=int(demo["patience"]),
            min_delta=float(demo["min_delta"]), num_threads=int(config["num_threads"]),
        )
        total_train_seconds += run["training_seconds"]
        validation_history[partition] = run["history"]
        pred = predecir_lgd(run["model"], x[valid_i], batch_size=int(demo["batch_size"]))
        observed = y[valid_i].astype(float)
        valid_ids = data.iloc[valid_i]["id"].to_numpy()
        row = {
            "modelo": "mlp_demo", "particion": partition, "conjunto": "validacion",
            "configuracion": model_config["id"], "semilla": int(demo["seed"]),
            "n_train": int(len(train_i)), "n_validacion": int(len(valid_i)),
            "epoca_mejor": int(run["best_epoch"]), "epocas_ejecutadas": int(run["epochs_run"]),
            "parametros_entrenables": int(run["parameters"]),
            "tiempo_entrenamiento_s": float(run["training_seconds"]),
            **metricas_lgd(observed, pred),
            "pred_media": float(pred.mean()), "lgd_observada_media": float(observed.mean()),
        }
        all_rows.append(row)
        pred_frames.append(pd.DataFrame({
            "id": valid_ids, "particion": partition, "conjunto": "validacion",
            "lgd_observada": observed, "lgd_predicha": pred,
        }))
        distribution_rows.append(resumen_distribucion(observed, observed, "observada", partition))
        distribution_rows.append(resumen_distribucion(pred, observed, "mlp_predicha", partition))

        baseline = ridge_metrics.loc[
            (ridge_metrics["particion"] == partition)
            & (ridge_metrics["conjunto"] == "validacion")
        ]
        for model_name in ("ridge", "media_train"):
            r = baseline.loc[baseline["modelo"] == model_name]
            if len(r) != 1:
                raise ValueError(f"Falta métrica baseline {model_name}/{partition}/validacion")
            b = r.iloc[0]
            all_rows.append({
                "modelo": model_name, "particion": partition, "conjunto": "validacion",
                "configuracion": "guardada_en_results_lgd_lineal_csv", "semilla": "determinista",
                "n_train": int(indices["entrenamiento"].size), "n_validacion": int(len(valid_i)),
                "epoca_mejor": None, "epocas_ejecutadas": None,
                "parametros_entrenables": int(b["params"]), "tiempo_entrenamiento_s": float(b["tiempo_busqueda_s"]),
                "mae": float(b["mae"]), "rmse": float(b["rmse"]), "sesgo": float(b["sesgo"]),
                "r2": float(b["r2"]), "pred_media": None, "lgd_observada_media": float(observed.mean()),
            })

        checkpoint = model_dir / f"lgd_mlp_{partition}_demo_seed_{demo['seed']}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": run["state_dict"], "input_dim": int(x.shape[1]),
            "input_columns": input_columns, "output_feature_count": int(x.shape[1]),
            "config": model_config, "seed": int(demo["seed"]), "partition": partition,
            "best_epoch": int(run["best_epoch"]), "epochs_run": int(run["epochs_run"]),
            "parameters": int(run["parameters"]), "hashes_input": hashes,
        }, checkpoint)

    metrics = pd.DataFrame(all_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    distribution = pd.DataFrame(distribution_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metricas_validacion_y_baselines.csv", index=False)
    predictions.to_parquet(out_dir / "predicciones_validacion.parquet", index=False)
    distribution.to_csv(out_dir / "distribucion_validacion.csv", index=False)
    with (out_dir / "historial_entrenamiento.json").open("w", encoding="utf-8") as f:
        json.dump(validation_history, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, partition in enumerate(PARTITIONS):
        g = predictions[predictions["particion"] == partition]
        obs = g["lgd_observada"].to_numpy()
        pred = g["lgd_predicha"].to_numpy()
        ax = axes[0, col]
        bins = np.linspace(0.0, 1.0, 21)
        ax.hist(obs, bins=bins, weights=np.ones(len(obs)) / len(obs), alpha=.55, label="Observada")
        ax.hist(pred, bins=bins, weights=np.ones(len(pred)) / len(pred), alpha=.55, label="MLP demo")
        ax.set(title=f"{partition}: validación", xlabel="LGD", ylabel="Proporción")
        ax.legend(frameon=False)
        ax = axes[1, col]
        q = np.linspace(.01, .99, 99)
        ax.plot(np.quantile(obs, q), np.quantile(pred, q), ".", ms=4)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Cuantiles LGD observada", ylabel="Cuantiles MLP")
        ax.set_title(f"Q-Q: {partition} validación")
    fig.tight_layout()
    fig.savefig(out_dir / "figura_distribucion_validacion.png", dpi=180)
    plt.close(fig)

    verification = json.loads((ROOT / "results/verificacion_datos_procesados.json").read_text(encoding="utf-8"))
    basis = verification.get("preprocesador_ajustado_con", "desconocido")
    summary = {
        "modo": "demo_integracion_no_final", "evaluacion": "solo se calcularon métricas de validación; la prueba no intervino en selección",
        "muestra": "df[df.castigado == True], igual a la muestra usada por Ridge",
        "filas_castigados": int(len(data)), "filas_por_particion": {
            p: {s: int(len(np.flatnonzero(data[c].to_numpy() == s))) for s in SETS}
            for p, c in PARTITIONS.items()
        },
        "ids_coinciden_con_ridge_en_train_valid_prueba": True,
        "caracteristicas_transformadas": int(x.shape[1]), "columnas_originales_preprocesador": input_columns,
        "configuracion": model_config, "loss": "MSE sobre objetivo continuo", "salida": "sigmoid en [0,1]",
        "semilla": int(demo["seed"]), "max_epocas": int(demo["max_epochs"]),
        "parametros_entrenables": int(numero_parametros(MLPRegresorLGD(x.shape[1], model_config))),
        "tiempo_entrenamiento_sumado_s": float(total_train_seconds),
        "tiempo_pared_s": float(time.perf_counter() - wall_start),
        "preprocesador_ajustado_con": basis,
        "limitacion_temporal_preprocesador": basis == "entrenamiento_aleatorio",
        "nota_temporal": "El preprocesador compartido se ajustó con entrenamiento aleatorio; puede haber visto filas temporales de validación/prueba.",
        "versiones": {"python": sys.version.split()[0], "torch": torch.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__},
        "hashes_entrada": hashes,
    }
    summary["parametros_entrenables_por_particion"] = {
        row["particion"]: row["parametros_entrenables"] for row in all_rows if row["modelo"] == "mlp_demo"
    }
    (out_dir / "resumen_mlp_lgd.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[metrics["conjunto"] == "validacion"].round(5).to_string(index=False))
    print(json.dumps({"output": str(out_dir), "models": str(model_dir), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
