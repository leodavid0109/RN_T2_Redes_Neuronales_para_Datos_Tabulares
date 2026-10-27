"""Piloto validation-only para comprobar convergencia/inicialización de LGD."""
from __future__ import annotations

import hashlib
import json
import argparse
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
from src.mlp_lgd import MLPRegresorLGD, entrenar_una_corrida, numero_parametros, predecir_lgd

OUT = ROOT / "results/lgd_mlp_convergencia_piloto_v2"
MODELS = ROOT / "models/lgd_mlp_convergencia_piloto_v2"
INPUTS = [
    "data/processed/lgd.parquet", "models/preprocessor.joblib",
    "results/lgd_lineal.csv", "results/lgd_lineal_pred.parquet",
    "results/verificacion_datos_procesados.json", "src/mlp_lgd.py",
    "configs/mlp_lgd_convergencia.yaml",
    "scripts/17_piloto_convergencia_mlp_lgd.py",
    "docs/mlp_lgd_convergencia.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT.relative_to(ROOT)))
    parser.add_argument("--models-dir", default=str(MODELS.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    model_dir = Path(args.models_dir)
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    if not model_dir.is_absolute(): model_dir = ROOT / model_dir
    for p in (out_dir, model_dir):
        if p.exists() and any(p.iterdir()):
            raise FileExistsError(f"No sobrescribo carpeta con contenido: {p}")
    config = yaml.safe_load((ROOT / "configs/mlp_lgd_convergencia.yaml").read_text(encoding="utf-8"))
    torch.set_num_threads(config["num_threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data = pd.read_parquet(ROOT / "data/processed/lgd.parquet")
    data = data.loc[data["castigado"].astype(bool)].copy().reset_index(drop=True)
    pre = joblib.load(ROOT / "models/preprocessor.joblib")
    columns = list(pre.feature_names_in_)
    x = np.asarray(pre.transform(data[columns]), dtype=np.float32)
    y = data["lgd"].to_numpy(dtype=np.float32)
    if x.shape[1] != 99 or not np.isfinite(x).all():
        raise ValueError(f"Matriz transformada inesperada/no finita: {x.shape}")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("LGD no finita o fuera de [0,1]")

    ridge = pd.read_csv(ROOT / "results/lgd_lineal.csv")
    predictions = pd.read_parquet(ROOT / "results/lgd_lineal_pred.parquet")
    hashes = {name: sha256(ROOT / name) for name in INPUTS}
    # Los IDs se contrastan solo para train/validación; no se inspeccionan métricas de prueba.
    rows, pred_rows, histories = [], [], {}
    elapsed_train = 0.0
    t0 = time.perf_counter()
    model_spec = config["model"]

    for partition, split_col in (("aleatoria", "split_random"), ("temporal", "split_temporal")):
        labels = data[split_col].to_numpy()
        train_i, valid_i = np.flatnonzero(labels == "entrenamiento"), np.flatnonzero(labels == "validacion")
        for split, idx in (("entrenamiento", train_i), ("validacion", valid_i)):
            expected = set(predictions.loc[
                (predictions.particion == partition) & (predictions.conjunto == split), "id"].tolist())
            actual = set(data.iloc[idx].id.tolist())
            if expected != actual:
                raise ValueError(f"IDs distintos de Ridge en {partition}/{split}")
        for init in config["initializations"]:
            for seed in config["seeds"]:
                run = entrenar_una_corrida(
                    x[train_i], y[train_i], x[valid_i], y[valid_i], model_spec,
                    seed=seed, batch_size=config["batch_size"], max_epochs=config["max_epochs"],
                    patience=config["patience"], min_delta=config["min_delta"],
                    num_threads=config["num_threads"], output_initialization=init,
                )
                elapsed_train += run["training_seconds"]
                pred = predecir_lgd(run["model"], x[valid_i], config["batch_size"])
                observed = y[valid_i].astype(float)
                key = f"{partition}/{init}/seed_{seed}"
                histories[key] = run["history"]
                metrics = metricas_lgd(observed, pred)
                rows.append({
                    "modelo": "mlp_lgd", "particion": partition, "inicializacion": init,
                    "semilla": seed, "arquitectura": "99-32-1", "activacion": "relu",
                    "salida": "sigmoid", "perdida": "mse", "optimizador": "adam",
                    "learning_rate": 0.001, "batch_size": config["batch_size"],
                    "max_epocas": config["max_epochs"], "paciencia": config["patience"],
                    "epoca_mejor": run["best_epoch"], "epocas_ejecutadas": run["epochs_run"],
                    "motivo_parada": run["stop_reason"], "parametros_entrenables": run["parameters"],
                    "tiempo_entrenamiento_s": run["training_seconds"],
                    "media_train_para_inicializacion": float(y[train_i].mean()),
                    "media_predicha": float(pred.mean()), "media_observada": float(observed.mean()),
                    **metrics,
                })
                pred_rows.append(pd.DataFrame({
                    "id": data.iloc[valid_i].id.to_numpy(), "particion": partition,
                    "inicializacion": init, "semilla": seed,
                    "lgd_observada": observed, "lgd_predicha": pred,
                }))
                ckpt = model_dir / f"{partition}_{init}_seed_{seed}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": run["state_dict"], "input_dim": int(x.shape[1]),
                    "input_columns": columns, "config": model_spec, "seed": seed,
                    "partition": partition, "initialization": init,
                    "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
                    "stop_reason": run["stop_reason"], "parameters": run["parameters"],
                    "train_lgd_mean": float(y[train_i].mean()), "hashes_input": hashes,
                }, ckpt)

    metrics_df = pd.DataFrame(rows)
    # Baselines: solo extraemos las filas de validación solicitadas; nunca leemos scores de test.
    baseline_rows = []
    for partition in ("aleatoria", "temporal"):
        for model_name in ("ridge", "media_train"):
            r = ridge.loc[(ridge.particion == partition) & (ridge.conjunto == "validacion") & (ridge.modelo == model_name)]
            if len(r) != 1:
                raise ValueError(f"No hay única referencia validación {partition}/{model_name}")
            rec = r.iloc[0]
            baseline_rows.append({"modelo": model_name, "particion": partition,
                                  "rmse": float(rec.rmse), "mae": float(rec.mae),
                                  "sesgo": float(rec.sesgo), "r2": float(rec.r2),
                                  "n": int(rec.n)})
    baselines = pd.DataFrame(baseline_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "metricas_validacion.csv", index=False)
    baselines.to_csv(out_dir / "baselines_validacion.csv", index=False)
    pred_df.to_parquet(out_dir / "predicciones_validacion.parquet", index=False)
    with (out_dir / "trayectorias_por_epoca.json").open("w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, partition in enumerate(("aleatoria", "temporal")):
        for init in config["initializations"]:
            for seed in config["seeds"]:
                h = histories[f"{partition}/{init}/seed_{seed}"]
                label = f"{init}, semilla {seed}"
                axes[0, col].plot([q["epoch"] for q in h], [q["train_mse_mean"] for q in h], label=label)
                axes[1, col].plot([q["epoch"] for q in h], [q["valid_rmse"] for q in h], label=label)
        axes[0, col].set(title=f"{partition}: pérdida train", xlabel="Época", ylabel="MSE")
        axes[1, col].set(title=f"{partition}: validación", xlabel="Época", ylabel="RMSE")
        axes[1, col].axhline(float(baselines.loc[(baselines.particion == partition) & (baselines.modelo == "ridge"), "rmse"].iloc[0]), color="black", ls="--", label="Ridge")
        axes[1, col].axhline(float(baselines.loc[(baselines.particion == partition) & (baselines.modelo == "media_train"), "rmse"].iloc[0]), color="gray", ls=":", label="Media train")
        axes[0, col].legend(fontsize=7); axes[1, col].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "curvas_convergencia_validacion.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 26)
    for ax, partition in zip(axes, ("aleatoria", "temporal")):
        predp = pred_df[pred_df.particion == partition]
        obs = predp.drop_duplicates("id").lgd_observada.to_numpy()
        ax.hist(obs, bins=bins, density=True, alpha=.55, label="Observada")
        for init in config["initializations"]:
            vals = predp.loc[predp.inicializacion == init, "lgd_predicha"].to_numpy()
            ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.5, label=f"MLP {init}")
        ax.set(title=f"{partition}: validación", xlabel="LGD", ylabel="Densidad")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "distribuciones_validacion.png", dpi=180); plt.close(fig)

    elapsed = time.perf_counter() - t0
    ver = json.loads((ROOT / "results/verificacion_datos_procesados.json").read_text(encoding="utf-8"))
    summary = {
        "estado": "piloto_convergencia_validation_only_no_final",
        "prueba": "no se calcularon metricas ni se usaron scores de prueba",
        "n_corridas": len(metrics_df), "n_semillas_por_condicion_y_particion": len(config["seeds"]),
        "configuracion": config, "metricas": metrics_df.to_dict(orient="records"),
        "baselines_validacion": baselines.to_dict(orient="records"),
        "tiempo_entrenamiento_s": elapsed_train, "tiempo_pared_s": elapsed,
        "presupuesto_previsto_s": config["budget_seconds"],
        "preprocesador_ajustado_con": ver.get("preprocesador_ajustado_con"),
        "limitacion_temporal_preprocesador": "El preprocesador compartido fue ajustado con entrenamiento aleatorio y pudo incorporar filas de validación temporal; persiste la limitación conocida.",
        "versiones": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": __import__("sklearn").__version__},
        "hashes_entrada": hashes,
        "hash_modelos": {p.name: sha256(p) for p in sorted(model_dir.glob("*.pt"))},
    }
    (out_dir / "resumen_piloto.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_corridas": len(metrics_df), "tiempo_entrenamiento_s": elapsed_train,
                      "tiempo_pared_s": elapsed, "resultados": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
