"""Búsqueda LGD validation-only, elegida por partición y reproducible."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/lgd_mlp_search_v1"
MODELS = ROOT / "models/lgd_mlp_search_v1"
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
from src.metricas_lgd import metricas_lgd
from src.mlp_lgd import MLPRegresorLGD, entrenar_una_corrida, numero_parametros, predecir_lgd

PARTITIONS = {"aleatoria": "split_random", "temporal": "split_temporal"}
SETS = ("entrenamiento", "validacion", "prueba")
INPUTS = (
    "data/processed/lgd.parquet", "models/preprocessor.joblib",
    "results/lgd_lineal.csv", "results/lgd_lineal_pred.parquet",
    "results/lgd_lineal_resumen.json", "results/verificacion_datos_procesados.json",
    "configs/mlp_lgd.yaml", "src/mlp_lgd.py", "src/metricas_lgd.py",
    "scripts/19_buscar_mlp_lgd.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_new(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"No sobrescribo resultados existentes: {path}")


def distribution_summary(values: np.ndarray, observed: np.ndarray, partition: str,
                          config_id: str, init: str, seed: int) -> dict:
    return {
        "particion": partition, "configuracion": config_id,
        "inicializacion": init, "semilla": int(seed), "n": int(len(values)),
        "media_predicha": float(np.mean(values)), "desv_predicha": float(np.std(values, ddof=1)),
        "mediana": float(np.median(values)), "p10": float(np.quantile(values, .10)),
        "p90": float(np.quantile(values, .90)), "min": float(np.min(values)),
        "max": float(np.max(values)), "fraccion_predicha_mayor_099": float(np.mean(values > .99)),
        "masa_predicha_1": float(np.mean(values == 1.0)),
        "media_observada": float(np.mean(observed)),
        "fraccion_observada_1": float(np.mean(observed == 1.0)),
        "fraccion_observada_0": float(np.mean(observed == 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=["search"], default="search")
    parser.add_argument("--config", default="configs/mlp_lgd.yaml")
    parser.add_argument("--output-dir", default="results/lgd_mlp_search_v1")
    parser.add_argument("--models-dir", default="models/lgd_mlp_search_v1")
    args = parser.parse_args()
    config_path = Path(args.config)
    out_dir, model_dir = Path(args.output_dir), Path(args.models_dir)
    if not config_path.is_absolute(): config_path = ROOT / config_path
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    if not model_dir.is_absolute(): model_dir = ROOT / model_dir
    ensure_new(out_dir); ensure_new(model_dir)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    search = config["search"]
    candidates = search["candidates"]
    if len(candidates) != 10:
        raise ValueError(f"Se esperan exactamente 10 candidatos declarados; hay {len(candidates)}")
    if len(set(c["id"] for c in candidates)) != len(candidates):
        raise ValueError("IDs de candidatos duplicados")
    if any(c.get("output_initialization") != "mean_train" for c in candidates if c["id"] != "base32_relu_adam_pytorch_init"):
        raise ValueError("Los nueve candidatos principales deben usar mean_train")
    torch.set_num_threads(int(config["num_threads"]))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    data = pd.read_parquet(ROOT / "data/processed/lgd.parquet")
    data = data.loc[data["castigado"].astype(bool)].copy().reset_index(drop=True)
    if data["id"].duplicated().any(): raise ValueError("La muestra castigada tiene IDs duplicados")
    pre = joblib.load(ROOT / "models/preprocessor.joblib")
    input_columns = list(pre.feature_names_in_)
    missing = sorted(set(input_columns).difference(data.columns))
    if missing: raise ValueError(f"Faltan columnas del preprocesador: {missing}")
    x = np.asarray(pre.transform(data[input_columns]), dtype=np.float32)
    y = data["lgd"].to_numpy(dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != int(config["input_features"]) or not np.isfinite(x).all():
        raise ValueError(f"Matriz transformada incompatible/no finita: {x.shape}")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("LGD inválida: se requiere objetivo finito en [0,1]")

    ridge_predictions = pd.read_parquet(ROOT / "results/lgd_lineal_pred.parquet")
    ridge_metrics = pd.read_csv(ROOT / "results/lgd_lineal.csv")
    indices: dict[str, dict[str, np.ndarray]] = {}
    id_check = {}
    # Igualdad estricta de muestras para Ridge en los tres conjuntos de cada partición.
    # Para prueba se comparan exclusivamente IDs; no se cargan objetivos/predicciones de prueba para métricas.
    for partition, split_column in PARTITIONS.items():
        labels = data[split_column].to_numpy()
        indices[partition] = {split: np.flatnonzero(labels == split) for split in SETS}
        id_check[partition] = {}
        for split, idx in indices[partition].items():
            ref_ids = set(ridge_predictions.loc[
                (ridge_predictions["particion"] == partition)
                & (ridge_predictions["conjunto"] == split), "id"].tolist())
            actual_ids = set(data.iloc[idx]["id"].tolist())
            if actual_ids != ref_ids:
                raise ValueError(f"IDs distintos de Ridge {partition}/{split}: {len(actual_ids)} vs {len(ref_ids)}")
            id_check[partition][split] = {"n": len(actual_ids), "coincide_con_ridge": True}

    hashes = {p: sha256(ROOT / p) for p in INPUTS}
    verification = json.loads((ROOT / "results/verificacion_datos_procesados.json").read_text(encoding="utf-8"))
    results: list[dict] = []
    distributions: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    histories: dict[str, list[dict]] = {}
    model_dir.mkdir(parents=True, exist_ok=True)
    start_wall = time.perf_counter()
    train_time = 0.0

    for partition in PARTITIONS:
        train_i, valid_i = indices[partition]["entrenamiento"], indices[partition]["validacion"]
        obs = y[valid_i].astype(float)
        for candidate in candidates:
            model_config = {k: candidate[k] for k in (
                "id", "hidden_layers", "activation", "dropout", "weight_decay", "optimizer", "learning_rate")}
            init = candidate["output_initialization"]
            for seed in search["seeds"]:
                run = entrenar_una_corrida(
                    x[train_i], y[train_i], x[valid_i], y[valid_i], model_config,
                    seed=int(seed), batch_size=int(search["batch_size"]),
                    max_epochs=int(search["max_epochs"]), patience=int(search["patience"]),
                    min_delta=float(search["min_delta"]), num_threads=int(config["num_threads"]),
                    output_initialization=init,
                )
                observed_best = min(item["valid_rmse"] for item in run["history"])
                if not np.isclose(run["best_validation_rmse"], observed_best, rtol=0, atol=1e-12):
                    raise AssertionError("El checkpoint no corresponde al mínimo RMSE observado en validación")
                train_time += float(run["training_seconds"])
                pred = predecir_lgd(run["model"], x[valid_i], batch_size=int(search["batch_size"]))
                metrics = metricas_lgd(obs, pred)
                row = {
                    "configuracion": candidate["id"], "cambio_vs_base": candidate["changed_from_base"],
                    "hipotesis": candidate["hypothesis"], "particion": partition,
                    "inicializacion": init, "semilla": int(seed),
                    "arquitectura": f"{x.shape[1]}-" + "-".join(map(str, candidate["hidden_layers"])) + "-1",
                    "capas_ocultas": json.dumps(candidate["hidden_layers"]),
                    "activacion": candidate["activation"], "dropout": float(candidate["dropout"]),
                    "weight_decay": float(candidate["weight_decay"]), "optimizador": candidate["optimizer"],
                    "learning_rate": float(candidate["learning_rate"]),
                    "salida": config["output_activation"], "perdida": config["loss"],
                    "batch_size": int(search["batch_size"]), "max_epocas": int(search["max_epochs"]),
                    "paciencia": int(search["patience"]), "min_delta": float(search["min_delta"]),
                    "epoca_mejor": int(run["best_epoch"]), "epocas_ejecutadas": int(run["epochs_run"]),
                    "motivo_parada": run["stop_reason"], "parametros_entrenables": int(run["parameters"]),
                    "tiempo_entrenamiento_s": float(run["training_seconds"]),
                    "media_train_inicializacion": float(y[train_i].mean()),
                    "media_predicha": float(pred.mean()), "media_observada": float(obs.mean()),
                    **metrics,
                }
                results.append(row)
                distributions.append(distribution_summary(pred, obs, partition, candidate["id"], init, int(seed)))
                histories[f"{partition}/{candidate['id']}/seed_{seed}"] = run["history"]
                pred_frames.append(pd.DataFrame({
                    "id": data.iloc[valid_i]["id"].to_numpy(), "particion": partition,
                    "configuracion": candidate["id"], "inicializacion": init,
                    "semilla": int(seed), "lgd_observada": obs, "lgd_predicha": pred,
                }))
                checkpoint = model_dir / f"{partition}_{candidate['id']}_seed_{seed}.pt"
                torch.save({
                    "state_dict": run["state_dict"], "input_dim": int(x.shape[1]),
                    "input_columns": input_columns, "config": model_config, "output_initialization": init,
                    "seed": int(seed), "partition": partition, "best_epoch": int(run["best_epoch"]),
                    "epochs_run": int(run["epochs_run"]), "stop_reason": run["stop_reason"],
                    "parameters": int(run["parameters"]), "train_lgd_mean": float(y[train_i].mean()),
                    "hashes_input": hashes,
                }, checkpoint)

    runs = pd.DataFrame(results)
    dist_df = pd.DataFrame(distributions)
    predictions = pd.concat(pred_frames, ignore_index=True)
    metric_cols = ["mae", "rmse", "sesgo", "r2", "media_predicha", "tiempo_entrenamiento_s",
                   "parametros_entrenables", "epocas_ejecutadas"]
    summary = runs.groupby(["particion", "configuracion", "inicializacion"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(str(p) for p in c if p) if isinstance(c, tuple) else c for c in summary.columns]
    summary = summary.reset_index(drop=True)
    selected = {}
    for partition in PARTITIONS:
        means = runs.loc[runs.particion == partition].groupby("configuracion")["rmse"].mean().sort_values()
        selected[partition] = {"configuracion": str(means.index[0]), "rmse_validacion_medio": float(means.iloc[0]),
                               "criterio": "menor media de RMSE de validación entre dos semillas; partición independiente"}
    baseline_rows = []
    for partition in PARTITIONS:
        for name in ("ridge", "media_train"):
            hit = ridge_metrics.loc[(ridge_metrics.particion == partition)
                                    & (ridge_metrics.conjunto == "validacion")
                                    & (ridge_metrics.modelo == name)]
            if len(hit) != 1: raise ValueError(f"Baseline ausente/duplicado: {partition}/{name}/validacion")
            b = hit.iloc[0]
            baseline_rows.append({"particion": partition, "modelo": name, "configuracion": "baseline_guardado",
                                  "rmse": float(b.rmse), "mae": float(b.mae), "sesgo": float(b.sesgo), "r2": float(b.r2)})
        for cid in ("base32_relu_adam", "base32_relu_adam_pytorch_init", selected[partition]["configuracion"]):
            group = runs.loc[(runs.particion == partition) & (runs.configuracion == cid)]
            baseline_rows.append({"particion": partition, "modelo": "mlp_search", "configuracion": cid,
                                  "rmse": float(group.rmse.mean()), "rmse_desv": float(group.rmse.std(ddof=1)),
                                  "mae": float(group.mae.mean()), "sesgo": float(group.sesgo.mean()),
                                  "r2": float(group.r2.mean())})

    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "metricas_por_corrida.csv", index=False)
    summary.to_csv(out_dir / "resumen_candidatos_media_desv.csv", index=False)
    dist_df.to_csv(out_dir / "distribuciones_predichas_validacion.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(out_dir / "comparacion_referencias_validacion.csv", index=False)
    predictions.to_parquet(out_dir / "predicciones_validacion.parquet", index=False)
    with (out_dir / "trayectorias_por_epoca.json").open("w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)

    # Una figura por partición: pérdida de entrenamiento y RMSE validación, cada candidato/semilla.
    for partition in PARTITIONS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for candidate in candidates:
            for seed in search["seeds"]:
                key = f"{partition}/{candidate['id']}/seed_{seed}"
                hist = histories[key]
                label = f"{candidate['id']} s{seed}"
                axes[0].plot([h["epoch"] for h in hist], [h["train_mse_mean"] for h in hist], alpha=.7, label=label)
                axes[1].plot([h["epoch"] for h in hist], [h["valid_rmse"] for h in hist], alpha=.7, label=label)
        axes[0].set(title=f"{partition}: entrenamiento", xlabel="Época", ylabel="MSE medio por lote")
        axes[1].set(title=f"{partition}: validación", xlabel="Época", ylabel="RMSE")
        for ax in axes: ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(out_dir / f"curvas_{partition}.png", dpi=170); plt.close(fig)

    # Distribución validación observada y predicciones agrupadas por candidato (ambas semillas).
    fig, axes = plt.subplots(1, 2, figsize=(15, 5)); bins = np.linspace(0, 1, 26)
    for ax, partition in zip(axes, PARTITIONS):
        part_preds = predictions.loc[predictions.particion == partition]
        obs = part_preds.drop_duplicates("id")["lgd_observada"].to_numpy()
        ax.hist(obs, bins=bins, density=True, color="black", alpha=.25, label="LGD observada")
        for candidate in candidates:
            vals = part_preds.loc[part_preds.configuracion == candidate["id"], "lgd_predicha"].to_numpy()
            ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.1, label=candidate["id"])
        ax.set(title=f"{partition}: distribución en validación", xlabel="LGD", ylabel="Densidad")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / "distribuciones_validacion.png", dpi=170); plt.close(fig)

    total_wall = time.perf_counter() - start_wall
    summary_json = {
        "mode": args.modo, "status": "exploratory_search_validation_only_no_final_30_seeds",
        "test_metrics_calculated": False, "test_used_for_selection": False,
        "id_match_ridge_all_sets": id_check,
        "sample": "lgd.parquet filtrado castigado=True, preprocessor.joblib existente, 99 características; mismas máscaras que Ridge",
        "config": config, "candidate_count": len(candidates), "run_count": len(runs),
        "selected_per_partition": selected,
        "stop_counts": runs.groupby(["particion", "motivo_parada"]).size().rename("corridas").reset_index().to_dict(orient="records"),
        "training_seconds_total": train_time, "wall_seconds_total": total_wall,
        "cpu_budget_seconds": int(search["budget_cpu_minutes"] * 60),
        "preprocessor_basis": verification.get("preprocesador_ajustado_con"),
        "temporal_preprocessor_limitation": "El preprocesador compartido fue ajustado con entrenamiento aleatorio; pudo incorporar filas temporales de validación/prueba. El resultado temporal conserva esta limitación.",
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": __import__("sklearn").__version__},
        "input_hashes": hashes,
        "checkpoint_hashes": {p.name: sha256(p) for p in sorted(model_dir.glob("*.pt"))},
    }
    (out_dir / "resumen_busqueda.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mode": args.modo, "runs": len(runs), "selected": selected,
                      "stops": summary_json["stop_counts"], "training_seconds": train_time,
                      "wall_seconds": total_wall, "results": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
