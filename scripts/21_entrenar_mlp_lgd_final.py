"""Fase final LGD: 30 semillas por partición, con prueba solo tras congelar corridas."""
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
from src.metricas_lgd import metricas_lgd
from src.mlp_lgd import entrenar_una_corrida, predecir_lgd

PARTITIONS = {"aleatoria": "split_random", "temporal": "split_temporal"}
SETS = ("entrenamiento", "validacion", "prueba")
SEARCH_ARTIFACTS = (
    "resumen_busqueda.json", "metricas_por_corrida.csv", "resumen_candidatos_media_desv.csv",
    "predicciones_validacion.parquet", "trayectorias_por_epoca.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_new(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"No sobrescribo artefactos: {path}; use otra ruta")


def preflight(search_dir: Path) -> dict:
    search_dir = search_dir.resolve()
    try:
        search_label = search_dir.relative_to(ROOT).as_posix()
    except ValueError:
        search_label = str(search_dir)
    search_summary_path = search_dir / "resumen_busqueda.json"
    if not search_summary_path.is_file():
        raise FileNotFoundError(f"Falta el resumen congelado de búsqueda: {search_summary_path}")
    search = json.loads(search_summary_path.read_text(encoding="utf-8"))
    if search.get("mode") != "search" or search.get("test_metrics_calculated") or search.get("test_used_for_selection"):
        raise ValueError("El manifiesto de búsqueda no acredita una selección solo con validación")
    mismatches = []
    for relative, expected in search["input_hashes"].items():
        path = ROOT / relative
        got = sha256(path) if path.is_file() else None
        if got != expected: mismatches.append({"path": relative, "expected": expected, "actual": got})
    if mismatches:
        raise RuntimeError(f"Cambió una entrada de la búsqueda; no inicio final: {mismatches}")

    search_file_hashes = {name: sha256(search_dir / name) for name in SEARCH_ARTIFACTS}
    runs = pd.read_csv(search_dir / "metricas_por_corrida.csv")
    if len(runs) != 40 or runs["particion"].nunique() != 2:
        raise ValueError(f"Número/estructura inesperada de corridas de búsqueda: {len(runs)}")
    cfg = yaml.safe_load((ROOT / "configs/mlp_lgd.yaml").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cfg["search"]["candidates"]}
    winners = {}
    for partition, chosen in search["selected_per_partition"].items():
        means = runs.loc[runs.particion == partition].groupby("configuracion")["rmse"].mean()
        actual_winner = str(means.idxmin())
        if actual_winner != chosen["configuracion"]:
            raise ValueError(f"Selección guardada no coincide con corridas de {partition}: {actual_winner}")
        cand = by_id[chosen["configuracion"]]
        expected_init = "mean_train"
        if cand.get("output_initialization") != expected_init:
            raise ValueError(f"Inicialización final inesperada para {partition}: {cand.get('output_initialization')}")
        winners[partition] = cand

    expected_seeds = list(range(30))
    if cfg.get("final", {}).get("seeds") != expected_seeds:
        raise ValueError("La configuración final no declara exactamente semillas 0–29")
    if set(expected_seeds).intersection(cfg["search"]["seeds"]):
        raise ValueError("Semillas finales se solapan con búsqueda exploratoria")

    data = pd.read_parquet(ROOT / "data/processed/lgd.parquet")
    data = data.loc[data["castigado"].astype(bool)].copy().reset_index(drop=True)
    if data["id"].duplicated().any(): raise ValueError("IDs duplicados en muestra castigada")
    ridge = pd.read_parquet(ROOT / "results/lgd_lineal_pred.parquet")
    id_checks = {}
    for partition, col in PARTITIONS.items():
        labels = data[col].to_numpy()
        id_checks[partition] = {}
        for subset in SETS:
            ids = set(data.loc[labels == subset, "id"].tolist())
            ridge_ids = set(ridge.loc[(ridge.particion == partition) & (ridge.conjunto == subset), "id"].tolist())
            if ids != ridge_ids:
                raise ValueError(f"IDs de muestra final no coinciden con Ridge: {partition}/{subset}")
            id_checks[partition][subset] = {"n": len(ids), "coincide_con_ridge": True}

    return {
        "search_directory": search_label,
        "verified_search_input_hashes": search["input_hashes"],
        "verified_search_output_hashes": search_file_hashes,
        "selected_search_winners": search["selected_per_partition"],
        "frozen_configurations": winners,
        "final_seeds": expected_seeds,
        "id_checks_against_ridge": id_checks,
        "search_wall_seconds": float(search["wall_seconds_total"]),
        "search_training_seconds": float(search["training_seconds_total"]),
        "search_candidate_count": int(search["candidate_count"]),
        "search_run_count": int(search["run_count"]),
        "hashes_validated_before_training": True,
    }


def summarize_distribution(observed: np.ndarray, predicted: np.ndarray, partition: str, subset: str) -> list[dict]:
    rows = []
    for label, values in (("observada", observed), ("prediccion_media_30_semillas_por_id", predicted)):
        rows.append({
            "particion": partition, "conjunto": subset, "serie": label, "n": int(len(values)),
            "media": float(np.mean(values)), "desv": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "p10": float(np.quantile(values, .10)),
            "p25": float(np.quantile(values, .25)), "p50": float(np.quantile(values, .50)),
            "p75": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .90)),
            "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)),
            "max": float(np.max(values)), "proporcion_exactamente_1": float(np.mean(values == 1.0)),
            "proporcion_exactamente_0": float(np.mean(values == 0.0)),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=["final"], default="final")
    parser.add_argument("--preflight-only", action="store_true", help="Verifica hashes/selección/IDs sin entrenar")
    parser.add_argument("--search-dir", default="results/lgd_mlp_search_v1",
                        help="Carpeta de búsqueda congelada cuyo resumen y hashes verificará el modo final")
    parser.add_argument("--output-dir", default="results/lgd_mlp_final_v2")
    parser.add_argument("--models-dir", default="models/lgd_mlp_final_v2")
    args = parser.parse_args()
    out_dir, model_dir = Path(args.output_dir), Path(args.models_dir)
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    if not model_dir.is_absolute(): model_dir = ROOT / model_dir
    search_dir = Path(args.search_dir)
    if not search_dir.is_absolute(): search_dir = ROOT / search_dir
    pre = preflight(search_dir)
    if args.preflight_only:
        print(json.dumps(pre, ensure_ascii=False, indent=2))
        return 0
    require_new(out_dir); require_new(model_dir)

    # Carga y transformación son idénticas al camino Ridge; test no entra a entrenar_una_corrida.
    data = pd.read_parquet(ROOT / "data/processed/lgd.parquet")
    data = data.loc[data["castigado"].astype(bool)].copy().reset_index(drop=True)
    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
    input_columns = list(preprocessor.feature_names_in_)
    x = np.asarray(preprocessor.transform(data[input_columns]), dtype=np.float32)
    y = data["lgd"].to_numpy(dtype=np.float32)
    if x.shape[1] != 99 or not np.isfinite(x).all(): raise ValueError(f"Matriz transformada inválida: {x.shape}")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)): raise ValueError("LGD fuera de [0,1]")
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    output_hashes = {name: sha256(search_dir / name) for name in SEARCH_ARTIFACTS}
    source_hashes = dict(pre["verified_search_input_hashes"])
    source_hashes.update({f"{pre['search_directory']}/{k}": v for k, v in output_hashes.items()})
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_hashes_y_configuracion.json").write_text(
        json.dumps(pre, ensure_ascii=False, indent=2), encoding="utf-8")

    config_file = yaml.safe_load((ROOT / "configs/mlp_lgd.yaml").read_text(encoding="utf-8"))
    search_cfg = config_file["search"]
    seeds = pre["final_seeds"]
    run_records: list[dict] = []
    model_records: list[dict] = []
    fitted_models: dict[tuple[str, int], object] = {}
    indices = {}
    total_train_seconds = 0.0
    wall_start = time.perf_counter()

    # Fase de ajuste: solo train/valid. No se predice ni puntúa prueba aquí.
    for partition, split_col in PARTITIONS.items():
        labels = data[split_col].to_numpy()
        indices[partition] = {s: np.flatnonzero(labels == s) for s in SETS}
        ti, vi = indices[partition]["entrenamiento"], indices[partition]["validacion"]
        candidate = pre["frozen_configurations"][partition]
        model_config = {k: candidate[k] for k in ("id", "hidden_layers", "activation", "dropout", "weight_decay", "optimizer", "learning_rate")}
        for seed in seeds:
            run = entrenar_una_corrida(
                x[ti], y[ti], x[vi], y[vi], model_config, seed=seed,
                batch_size=int(search_cfg["batch_size"]), max_epochs=int(search_cfg["max_epochs"]),
                patience=int(search_cfg["patience"]), min_delta=float(search_cfg["min_delta"]),
                num_threads=2, output_initialization=candidate["output_initialization"],
            )
            observed_best = min(h["valid_rmse"] for h in run["history"])
            if not np.isclose(run["best_validation_rmse"], observed_best, rtol=0, atol=1e-12):
                raise AssertionError("El checkpoint final no corresponde al RMSE mínimo de validación")
            total_train_seconds += float(run["training_seconds"])
            fitted_models[(partition, seed)] = run["model"]
            record = {
                "particion": partition, "configuracion": candidate["id"],
                "inicializacion": candidate["output_initialization"], "semilla": seed,
                "arquitectura": f"99-{'-'.join(map(str,candidate['hidden_layers']))}-1",
                "capas_ocultas": candidate["hidden_layers"], "activacion": candidate["activation"],
                "salida": "sigmoid", "perdida": "mse", "optimizador": candidate["optimizer"],
                "learning_rate": float(candidate["learning_rate"]), "dropout": float(candidate["dropout"]),
                "weight_decay": float(candidate["weight_decay"]),
                "batch_size": int(search_cfg["batch_size"]), "max_epocas": int(search_cfg["max_epochs"]),
                "paciencia": int(search_cfg["patience"]), "min_delta": float(search_cfg["min_delta"]),
                "epoca_mejor": int(run["best_epoch"]), "epocas_ejecutadas": int(run["epochs_run"]),
                "motivo_parada": run["stop_reason"], "parametros_entrenables": int(run["parameters"]),
                "tiempo_entrenamiento_s": float(run["training_seconds"]),
                "media_train_inicializacion": float(y[ti].mean()),
            }
            run_records.append(record)
            checkpoint = model_dir / f"lgd_mlp_{partition}_seed_{seed:02d}.pt"
            torch.save({
                "state_dict": run["state_dict"], "input_dim": int(x.shape[1]),
                "input_columns": input_columns, "contract": "misma transformación/columnas que preprocessor.joblib y Ridge",
                "config": model_config, "output_initialization": candidate["output_initialization"],
                "partition": partition, "seed": seed, "best_epoch": int(run["best_epoch"]),
                "epochs_run": int(run["epochs_run"]), "stop_reason": run["stop_reason"],
                "parameters": int(run["parameters"]), "training_seconds": float(run["training_seconds"]),
                "train_lgd_mean": float(y[ti].mean()),
                "hashes_input": source_hashes,
            }, checkpoint)
            model_records.append({"particion": partition, "semilla": seed,
                                 "archivo": str(checkpoint.relative_to(ROOT)),
                                 "sha256": sha256(checkpoint), **record})

    # Todos los estados quedan congelados antes de la primera evaluación sobre test.
    pred_frames: list[pd.DataFrame] = []
    metric_records: list[dict] = []
    distribution_rows: list[dict] = []
    for partition in PARTITIONS:
        for subset in ("validacion", "prueba"):
            idx = indices[partition][subset]
            obs = y[idx].astype(float)
            ids = data.iloc[idx]["id"].to_numpy()
            seed_predictions = []
            for seed in seeds:
                pred = predecir_lgd(fitted_models[(partition, seed)], x[idx], batch_size=search_cfg["batch_size"])
                seed_predictions.append(pred)
                metric_records.append({
                    "particion": partition, "conjunto": subset, "configuracion": pre["frozen_configurations"][partition]["id"],
                    "semilla": seed, **metricas_lgd(obs, pred), "media_predicha": float(pred.mean()),
                    "media_observada": float(obs.mean()),
                })
                pred_frames.append(pd.DataFrame({
                    "id": ids, "particion": partition, "conjunto": subset,
                    "configuracion": pre["frozen_configurations"][partition]["id"],
                    "semilla": seed, "lgd_observada": obs, "lgd_predicha": pred,
                }))
            avg_pred = np.mean(np.stack(seed_predictions), axis=0)
            distribution_rows.extend(summarize_distribution(obs, avg_pred, partition, subset))

    # Baselines se alinean por ID en exactamente las mismas filas evaluadas por las 30 corridas.
    ridge_predictions = pd.read_parquet(ROOT / "results/lgd_lineal_pred.parquet")
    baselines = []
    for partition in PARTITIONS:
        train_mean = float(y[indices[partition]["entrenamiento"]].mean())
        for subset in ("validacion", "prueba"):
            idx = indices[partition][subset]
            ids = data.iloc[idx]["id"].to_numpy()
            obs = y[idx].astype(float)
            rr = ridge_predictions.loc[(ridge_predictions.particion == partition)
                                       & (ridge_predictions.conjunto == subset)].set_index("id").loc[ids]
            # LGD parquet was transformed to float32 for torch; allow only this storage-rounding delta.
            if not np.allclose(rr["lgd_obs"].to_numpy(dtype=float), obs, rtol=0, atol=1e-6):
                raise ValueError(f"LGD observada Ridge/data no alineada en {partition}/{subset}")
            for name, pred in (("ridge", rr["lgd_pred"].to_numpy(dtype=float)),
                               ("media_train", np.full(len(obs), train_mean, dtype=float))):
                baselines.append({"particion": partition, "conjunto": subset, "modelo": name,
                                  "media_train": train_mean if name == "media_train" else None,
                                  **metricas_lgd(obs, pred)})

    metrics_df = pd.DataFrame(metric_records)
    metrics_summary = []
    directions = {"mae": "min", "rmse": "min", "sesgo": "zero", "r2": "max"}
    for (partition, subset), group in metrics_df.groupby(["particion", "conjunto"]):
        for metric, direction in directions.items():
            values = group[metric]
            min_i, max_i = values.idxmin(), values.idxmax()
            if direction == "min": best_i, worst_i = min_i, max_i
            elif direction == "max": best_i, worst_i = max_i, min_i
            else:
                best_i = (group[metric].abs()).idxmin(); worst_i = (group[metric].abs()).idxmax()
            metrics_summary.append({
                "particion": partition, "conjunto": subset, "metrica": metric,
                "media": float(values.mean()), "desv_std": float(values.std(ddof=1)),
                "minimo": float(values.min()), "maximo": float(values.max()),
                "mejor_valor": float(group.loc[best_i, metric]), "mejor_semilla": int(group.loc[best_i, "semilla"]),
                "peor_valor": float(group.loc[worst_i, metric]), "peor_semilla": int(group.loc[worst_i, "semilla"]),
                "criterio_mejor_peor": "menor/mayor" if direction == "min" else "mayor/menor" if direction == "max" else "menor/mayor valor absoluto",
            })

    # Summary compare MLP across seeds against deterministic Ridge and mean baselines.
    baseline_df = pd.DataFrame(baselines)
    comparisons = []
    for partition in PARTITIONS:
        for subset in ("validacion", "prueba"):
            mlp = metrics_df.loc[(metrics_df.particion == partition) & (metrics_df.conjunto == subset)]
            comparisons.append({"particion": partition, "conjunto": subset, "modelo": "mlp_30_semillas",
                **{f"{m}_media": float(mlp[m].mean()) for m in directions},
                **{f"{m}_desv": float(mlp[m].std(ddof=1)) for m in directions}})
            for _, b in baseline_df.loc[(baseline_df.particion == partition) & (baseline_df.conjunto == subset)].iterrows():
                comparisons.append({"particion": partition, "conjunto": subset, "modelo": b["modelo"],
                    **{f"{m}_media": float(b[m]) for m in directions}, **{f"{m}_desv": 0.0 for m in directions}})

    # Figures use test only after the frozen 60 models have been evaluated, with per-ID mean over seeds.
    all_predictions = pd.concat(pred_frames, ignore_index=True)
    for partition in PARTITIONS:
        g = all_predictions.loc[(all_predictions.particion == partition) & (all_predictions.conjunto == "prueba")]
        avg = g.groupby("id", as_index=False).agg(lgd_observada=("lgd_observada", "first"), lgd_predicha=("lgd_predicha", "mean"))
        fig, ax = plt.subplots(figsize=(8.5, 5))
        bins = np.linspace(0, 1, 26)
        ax.hist(avg.lgd_observada, bins=bins, density=True, color="#475569", alpha=.34, label="LGD observada (test)")
        ax.hist(avg.lgd_predicha, bins=bins, density=True, histtype="step", color="#dc2626", lw=2,
                label="MLP: media de 30 predicciones por crédito")
        obs_one = float((avg.lgd_observada == 1).mean()); pred_one = float((avg.lgd_predicha == 1).mean())
        ax.set(title=f"LGD {partition}: distribución en test", xlabel="LGD", ylabel="Densidad")
        ax.text(.02, .97, f"observada=1: {obs_one:.1%} | MLP media=1: {pred_one:.1%}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(out_dir / f"distribucion_test_{partition}.png", dpi=180); plt.close(fig)

    # Outputs are written only after the frozen runs and test evaluation are complete.
    run_df = pd.DataFrame(run_records)
    pred_df = all_predictions
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model_records).to_csv(out_dir / "corridas_modelos.csv", index=False)
    metrics_df.to_csv(out_dir / "metricas_por_semilla_validacion_prueba.csv", index=False)
    pd.DataFrame(metrics_summary).to_csv(out_dir / "resumen_metricas_30_semillas.csv", index=False)
    baseline_df.to_csv(out_dir / "metricas_baselines.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out_dir / "comparacion_mlp_ridge_media.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(out_dir / "cuantiles_y_masa_en_uno.csv", index=False)
    pred_df.to_parquet(out_dir / "predicciones_validacion_prueba.parquet", index=False)
    by_seed_time = run_df.groupby("particion")["tiempo_entrenamiento_s"].agg(["count", "sum", "mean", "std", "min", "max"]).reset_index()
    by_seed_time.to_csv(out_dir / "tiempos_por_particion.csv", index=False)

    ver = json.loads((ROOT / "results/verificacion_datos_procesados.json").read_text(encoding="utf-8"))
    search_summary = json.loads((search_dir / "resumen_busqueda.json").read_text(encoding="utf-8"))
    search_extension_path = ROOT / "results/lgd_mlp_search_extension_v2/resumen_extension.json"
    extension_wall = None
    if pre["search_directory"] == "results/lgd_mlp_search_v1" and search_extension_path.exists():
        extension_wall = json.loads(search_extension_path.read_text(encoding="utf-8"))["wall_seconds"]
    final_wall = time.perf_counter() - wall_start
    final_train_total = float(run_df.tiempo_entrenamiento_s.sum())
    summary = {
        "status": "final_30_seeds_per_partition_test_evaluated_after_freeze",
        "test_metrics_calculated_after_all_60_checkpoints_frozen": True,
        "candidate_count_explored": int(pre["search_candidate_count"]), "search_runs": int(pre["search_run_count"]),
        "final_runs": int(len(run_df)), "seeds": seeds,
        "frozen_configurations": {p: pre["frozen_configurations"][p] for p in PARTITIONS},
        "preflight": pre,
        "id_match_ridge_all_sets": pre["id_checks_against_ridge"],
        "sample": "lgd.parquet castigado=True; same preprocessor and split masks as Ridge; 99 transformed features",
        "metric_summary": metrics_summary, "baselines": baselines,
        "time": {
            "search_wall_seconds": float(search_summary["wall_seconds_total"]),
            "search_training_seconds": float(search_summary["training_seconds_total"]),
            "final_wall_seconds": float(final_wall), "final_training_seconds": final_train_total,
            "search_plus_final_wall_seconds": float(search_summary["wall_seconds_total"] + final_wall),
            "search_plus_final_training_seconds": float(search_summary["training_seconds_total"] + final_train_total),
            "successful_initialization_extension_wall_seconds_separate": float(extension_wall),
        },
        "stop_counts": run_df.groupby(["particion", "motivo_parada"]).size().rename("corridas").reset_index().to_dict(orient="records"),
        "parameters_by_partition": run_df.groupby("particion").parametros_entrenables.first().to_dict(),
        "preprocessor_basis": ver.get("preprocesador_ajustado_con"),
        "temporal_limitations": "Preprocesador compartido ajustado con entrenamiento aleatorio; pudo incorporar filas temporales de validación/prueba. LGD observada cambia entre periodos; no atribuirlo al MLP ni afirmar ausencia de fuga.",
        "distribution_summary": distribution_rows,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": __import__("sklearn").__version__},
        "input_hashes": source_hashes,
        "final_checkpoint_hashes": {r["archivo"]: r["sha256"] for r in model_records},
    }
    (out_dir / "resumen_final.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(run_df), "selected": {p: pre["frozen_configurations"][p]["id"] for p in PARTITIONS},
                      "final_wall_seconds": final_wall, "final_training_seconds": final_train_total,
                      "search_plus_final_wall_seconds": summary["time"]["search_plus_final_wall_seconds"],
                      "stops": summary["stop_counts"], "output": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
