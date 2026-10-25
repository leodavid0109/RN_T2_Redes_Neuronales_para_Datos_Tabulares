"""Análisis de sensibilidad pareado de BCEWithLogitsLoss(pos_weight) para PD.

La configuración seleccionada sin ponderación no cambia. La selección del análisis
usa únicamente log-loss de validación; la prueba se calcula aparte y no interviene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".venv" / "matplotlib-config"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metricas import curva_calibracion, metricas_pd  # noqa: E402
from src.mlp_pd import MLPBinario, entrenar_una_corrida, probabilidades_clase_1, numero_parametros  # noqa: E402

PARTITIONS = {"aleatoria": "split_random", "temporal": "split_temporal"}
SPLITS = ("validacion", "prueba")
METRIC_NAMES = ("auc", "gini", "ks", "brier", "logloss")
HASH_PATHS = (
    "data/processed/pd.parquet", "models/preprocessor.joblib",
    "models/pd_logreg_aleatoria.joblib", "models/pd_logreg_temporal.joblib",
    "results/pd_mlp_search_independiente_v1/pd_mlp_resumen.json",
    "results/pd_mlp_final_particionado/pd_mlp_resumen.json",
    "configs/mlp_pd.yaml", "configs/mlp_pd_ponderacion.yaml", "scripts/15_experimento_ponderacion_pd.py",
    "src/mlp_pd.py", "src/metricas.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def no_sobrescribir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"No sobrescribo salidas existentes: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/pd_mlp_ponderacion_sensibilidad_v1")
    parser.add_argument("--models-dir", default="models/pd_mlp_ponderacion_sensibilidad_v1")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    model_dir = Path(args.models_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    no_sobrescribir(out_dir)
    no_sobrescribir(model_dir)

    protocol_path = ROOT / "configs/mlp_pd_ponderacion.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    final_summary = json.loads((ROOT / "results/pd_mlp_final_particionado/pd_mlp_resumen.json").read_text(encoding="utf-8"))
    selected = final_summary["configuraciones_elegidas"]
    if final_summary.get("protocolo_seleccion") != "independiente_por_particion":
        raise ValueError("Los modelos de referencia no documentan selección independiente por partición")
    for part in PARTITIONS:
        cfg = selected[part]
        if (cfg.get("id") != "width64" or cfg.get("hidden_layers") != [64]
                or cfg.get("activation") != "relu" or cfg.get("optimizer") != "adam"
                or cfg.get("learning_rate") != 0.001 or cfg.get("dropout") != 0.0
                or cfg.get("weight_decay") != 0.0):
            raise ValueError(f"La referencia de {part} ya no es la arquitectura acordada width64")

    torch.set_num_threads(int(protocol["num_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    df = pd.read_parquet(ROOT / "data/processed/pd.parquet")
    y = df["objetivo"].to_numpy(dtype=np.int64)
    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
    input_columns = list(preprocessor.feature_names_in_)
    x = np.asarray(preprocessor.transform(df[input_columns]), dtype=np.float32)
    if x.shape[1] != 99 or not np.isfinite(x).all():
        raise ValueError("La transformación compartida debe producir 99 columnas numéricas finitas")
    hashes = {p: sha256(ROOT / p) for p in HASH_PATHS}

    records: list[dict] = []
    calibration: list[pd.DataFrame] = []
    prevalence: dict[str, dict] = {}
    total_fit_seconds = 0.0
    started = time.perf_counter()
    for partition, mask_col in PARTITIONS.items():
        masks = df[mask_col].to_numpy()
        train_mask = masks == "entrenamiento"
        y_train = y[train_mask]
        n0, n1 = int((y_train == 0).sum()), int((y_train == 1).sum())
        ratio = n0 / n1
        prevalence[partition] = {
            "train_n0": n0, "train_n1": n1, "train_prevalencia": float(y_train.mean()),
            "ratio_n0_div_n1": ratio,
            "validacion_prevalencia": float(y[masks == "validacion"].mean()),
            "prueba_prevalencia": float(y[masks == "prueba"].mean()),
        }
        split_data = {
            name: (x[masks == name], y[masks == name])
            for name in ("entrenamiento", "validacion", "prueba")
        }
        for condition in protocol["conditions"]:
            pos_weight = ratio if condition["weight"] == "train_only_n0_div_n1" else float(condition["weight"])
            for seed in protocol["seeds"]:
                run = entrenar_una_corrida(
                    *split_data["entrenamiento"], *split_data["validacion"], selected[partition],
                    seed=int(seed), batch_size=int(protocol["batch_size"]),
                    max_epochs=int(protocol["max_epochs"]), patience=int(protocol["patience"]),
                    min_delta=float(protocol["min_delta"]), num_threads=int(protocol["num_threads"]),
                    pos_weight=pos_weight,
                )
                total_fit_seconds += run["training_seconds"]
                checkpoint = model_dir / f"{partition}__{condition['id']}__seed_{seed}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": run["state_dict"], "input_dim": x.shape[1],
                    "input_columns": input_columns, "config": selected[partition],
                    "condition": condition["id"], "pos_weight": pos_weight,
                    "seed": int(seed), "partition": partition,
                    "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
                    "parameters": run["parameters"], "hashes_input": hashes,
                }, checkpoint)
                for split_name in SPLITS:
                    x_eval, y_eval = split_data[split_name]
                    probs = probabilidades_clase_1(run["model"], x_eval, batch_size=int(protocol["batch_size"]))
                    row = {
                        "condicion": condition["id"], "hipotesis": condition["hypothesis"],
                        "particion": partition, "conjunto": split_name, "semilla": int(seed),
                        "configuracion": "width64", "configuracion_json": json.dumps(selected[partition], sort_keys=True),
                        "pos_weight": pos_weight, "pos_weight_origen": "solo entrenamiento" if condition["weight"] == "train_only_n0_div_n1" else "fijo predeclarado",
                        "epoca_mejor": run["best_epoch"], "epocas_ejecutadas": run["epochs_run"],
                        "tiempo_entrenamiento_s": run["training_seconds"], "parametros_entrenables": run["parameters"],
                        **metricas_pd(y_eval, probs), "probabilidad_media_predicha": float(probs.mean()),
                        "tasa_observada": float(y_eval.mean()), "diferencia_pd_media_menos_tasa": float(probs.mean() - y_eval.mean()),
                    }
                    records.append(row)
                    cal = curva_calibracion(y_eval, probs, n_grupos=10)
                    cal.insert(0, "semilla", int(seed))
                    cal.insert(0, "conjunto", split_name)
                    cal.insert(0, "particion", partition)
                    cal.insert(0, "condicion", condition["id"])
                    cal.insert(0, "pos_weight", pos_weight)
                    calibration.append(cal)

    metrics = pd.DataFrame(records)
    # Medias, dispersión entre semillas y diferencias emparejadas por semilla frente al control.
    metric_columns = [*METRIC_NAMES, "probabilidad_media_predicha", "tasa_observada", "diferencia_pd_media_menos_tasa"]
    group_cols = ["condicion", "particion", "conjunto"]
    summary = metrics.groupby(group_cols)[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(v) for v in col if v) if isinstance(col, tuple) else col for col in summary.columns]
    control = metrics[metrics["condicion"] == "control_pos_weight_1"].set_index(["particion", "conjunto", "semilla"])
    deltas = []
    for _, row in metrics.iterrows():
        base = control.loc[(row["particion"], row["conjunto"], row["semilla"])]
        for metric in metric_columns:
            deltas.append({"condicion": row["condicion"], "particion": row["particion"],
                           "conjunto": row["conjunto"], "semilla": row["semilla"],
                           "metrica": metric, "diferencia_vs_control": float(row[metric] - base[metric])})
    delta_df = pd.DataFrame(deltas)
    delta_summary = delta_df.groupby(["condicion", "particion", "conjunto", "metrica"])["diferencia_vs_control"].agg(["mean", "std"]).reset_index()

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metricas_ponderacion_por_semilla.csv", index=False)
    pd.concat(calibration, ignore_index=True).to_csv(out_dir / "curva_calibracion_ponderacion.csv", index=False)
    summary.to_csv(out_dir / "resumen_ponderacion_media_desviacion.csv", index=False)
    delta_df.to_csv(out_dir / "diferencias_pareadas_vs_control_por_semilla.csv", index=False)
    delta_summary.to_csv(out_dir / "resumen_diferencias_pareadas_vs_control.csv", index=False)

    # Figura de log-loss de validación; prueba se excluye deliberadamente.
    val = metrics[metrics["conjunto"] == "validacion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, part in zip(axes, PARTITIONS):
        subset = val[val["particion"] == part]
        order = [c["id"] for c in protocol["conditions"]]
        vals = [subset.loc[subset["condicion"] == cond, "logloss"].to_numpy() for cond in order]
        ax.boxplot(vals, tick_labels=order, showmeans=True)
        ax.set_title(f"{part}: log-loss de validación")
        ax.set_ylabel("Log-loss (menor es mejor)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figura_logloss_validacion.png", dpi=180)
    plt.close(fig)

    cal_df = pd.concat(calibration, ignore_index=True)
    cal_df = cal_df[cal_df["conjunto"] == "validacion"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    for ax, part in zip(axes[0], PARTITIONS):
        for condition in protocol["conditions"]:
            c = cal_df[(cal_df["particion"] == part) & (cal_df["condicion"] == condition["id"])]
            g = c.groupby("grupo").agg(pred=("prob_predicha_media", "mean"), obs=("tasa_observada", "mean"))
            ax.plot(g["pred"], g["obs"], marker="o", label=condition["id"])
        hi = float(max(ax.get_xlim()[1], ax.get_ylim()[1], cal_df[cal_df["particion"] == part]["tasa_observada"].max()))
        hi = max(hi, 0.01)
        ax.plot([0, hi], [0, hi], "--", color="gray", label="ideal")
        ax.set(xlabel="PD media predicha por decil", ylabel="Tasa observada", title=f"{part}: validación", xlim=(0, hi), ylim=(0, hi))
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "figura_calibracion_validacion.png", dpi=180)
    plt.close(fig)

    summary_json = {
        "modo": "analisis_sensibilidad_ponderacion_posterior",
        "no_es_nueva_seleccion_final_imparcial": True,
        "criterio_uso": "comparación primaria solo con log-loss de validación; prueba se reporta como sensibilidad y no se usa para elegir",
        "arquitectura": "width64 seleccionada previamente sin ponderación",
        "preprocesador": final_summary["preprocesador"],
        "hipotesis_y_condiciones": protocol["conditions"],
        "semillas_pareadas": protocol["seeds"],
        "numero_corridas": len(protocol["conditions"]) * len(protocol["seeds"]) * len(PARTITIONS),
        "max_epocas_por_corrida": protocol["max_epochs"],
        "presupuesto_cpu_minutos": protocol["budget_cpu_minutes"],
        "conteos_y_pesos_solo_train": prevalence,
        "tiempo_suma_entrenamiento_s": total_fit_seconds,
        "tiempo_pared_s": time.perf_counter() - started,
        "class_weight": "sin remuestreo; pos_weight en BCEWithLogitsLoss solo para train; parada evalúa log-loss no ponderada en validación",
        "decision": "el control pos_weight=1 conserva el modelo principal; tratamientos son análisis de sensibilidad",
        "versiones": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "hashes_entrada": hashes,
    }
    (out_dir / "resumen_ponderacion.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"salida": str(out_dir), "modelos": str(model_dir), "corridas": summary_json["numero_corridas"],
                      "tiempo_entrenamiento_s": total_fit_seconds, "tiempo_pared_s": summary_json["tiempo_pared_s"],
                      "weights": prevalence}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
