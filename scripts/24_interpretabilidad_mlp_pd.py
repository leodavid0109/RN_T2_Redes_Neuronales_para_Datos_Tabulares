"""Importancia por permutación agrupada del MLP de PD, solo en validación.

Usa el ensamble de 30 checkpoints publicados por partición. Las columnas one-hot de
cada variable categórica se permutan juntas; cada indicador de faltante se mueve con
su variable numérica. La importancia principal es el aumento de log-loss al romper
una variable. No es causal ni mide un valor intrínseco de la variable.

Ejemplo:
  python scripts/24_interpretabilidad_mlp_pd.py --output-dir results/pd_mlp_interpretabilidad_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv" / "matplotlib-config"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.mlp_pd import MLPBinario, probabilidades_clase_1

OUT_DEFAULT = "results/pd_mlp_interpretabilidad_v2"
SEED_REPEAT = 20261027
N_REPEATS = 5
SEEDS = range(30)
MISSING_PAIRS = {
    "mths_since_last_delinq": "falta_mths_since_last_delinq",
    "mths_since_last_record": "falta_mths_since_last_record",
    "tot_coll_amt": "falta_tot_coll_amt",
    "tot_cur_bal": "falta_tot_cur_bal",
    "total_rev_hi_lim": "falta_total_rev_hi_lim",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    auc = float(roc_auc_score(y, p))
    return {"auc": auc, "gini": 2 * auc - 1,
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, p, labels=[0, 1]))}


def grupos_columnas(names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for base, flag in MISSING_PAIRS.items():
        groups[base] = [names.index(base), names.index(flag)]
    paired = {c for pair in MISSING_PAIRS.items() for c in pair}
    prefixes = {"home_ownership": "home_ownership", "verification_status": "verification_status",
                "purpose": "purpose", "addr_state": "addr_state"}
    for name in names:
        if name in paired:
            continue
        prefix = next((p for p in prefixes if name.startswith(p + "_")), None)
        if prefix:
            groups.setdefault(prefix, []).append(names.index(name))
        else:
            groups[name] = [names.index(name)]
    if len(groups) != 24:
        raise ValueError(f"Se esperaban 24 grupos semánticos; se encontraron {len(groups)}: {list(groups)}")
    return groups


def load_ensemble(pre, part: str):
    models = []
    for seed in SEEDS:
        path = ROOT / "models" / "pd_mlp_final_particionado" / f"pd_mlp_{part}_final_seed_{seed}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["input_dim"] != len(pre.get_feature_names_out()) or list(ck["input_columns"]) != list(pre.feature_names_in_):
            raise ValueError(f"Contrato de entradas distinto en {path}")
        model = MLPBinario(ck["input_dim"], ck["model_config"])
        model.load_state_dict(ck["state_dict"])
        models.append(model)
    return models


def predict_ensemble(X: np.ndarray, models, batch: int = 8192) -> np.ndarray:
    total = np.zeros(len(X), dtype=np.float64)
    for model in models:
        total += probabilidades_clase_1(model, X, batch_size=batch)
    return total / len(models)


def write_plot(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=False)
    for ax, part in zip(axes, ("aleatoria", "temporal")):
        data = summary[summary.particion == part].sort_values("delta_logloss")
        ax.barh(data.variable, data.delta_logloss, xerr=data.delta_logloss_sd,
                color="#0072B2", alpha=.85, capsize=2)
        ax.axvline(0, color="#444", linewidth=.8)
        ax.set_title(f"{part.capitalize()} — validación")
        ax.set_xlabel("Aumento de log-loss al permutar (mayor = más importante)")
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def hypothesis_groups(pdf: pd.DataFrame, part: str, cols: list[str], p: np.ndarray) -> pd.DataFrame:
    """Tablas asociativas con cortes definidos a partir de las filas de validación."""
    split_col = "split_random" if part == "aleatoria" else "split_temporal"
    d = pdf.loc[pdf[split_col].eq("validacion")].copy().reset_index(drop=True)
    d["pd_media_ensamble"] = p
    specs = {
        "H1_dti": ("dti", lambda s: pd.qcut(s, 5, duplicates="drop")),
        "H2_term": ("term_meses", lambda s: s.astype(str)),
        "H3_revol_util": ("revol_util", lambda s: pd.qcut(s, 5, duplicates="drop")),
        "H4_delinq_2yrs": ("delinq_2yrs", lambda s: np.where(s >= 1, "1+", "0")),
        "H5_inq_last_6mths": ("inq_last_6mths", lambda s: np.select([s == 0, s == 1, s == 2], ["0", "1", "2"], default="3+")),
        "H6_annual_inc": ("annual_inc", lambda s: pd.qcut(s, 5, duplicates="drop")),
        "H7_antiguedad_cred_meses": ("antiguedad_cred_meses", lambda s: pd.qcut(s, 5, duplicates="drop")),
        "H8_home_ownership": ("home_ownership", lambda s: s.astype(str)),
    }
    rows = []
    for h, (column, cut) in specs.items():
        d["grupo"] = cut(d[column])
        for group, frame in d.groupby("grupo", observed=True, dropna=False):
            rows.append({"particion": part, "hipotesis": h, "variable": column,
                         "grupo": str(group), "n": int(len(frame)),
                         "tasa_incumplimiento_observada": float(frame["objetivo"].mean()),
                         "pd_media_predicha_ensamble": float(frame["pd_media_ensamble"].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=OUT_DEFAULT)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--num-threads", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"La carpeta de salida no está vacía: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    t0 = time.perf_counter()
    pdf_path, pre_path = ROOT / "data/processed/pd.parquet", ROOT / "models/preprocessor.joblib"
    pdf = pd.read_parquet(pdf_path)
    pre = joblib.load(pre_path)
    cols = list(pre.feature_names_in_)
    transformed_names = list(pre.get_feature_names_out())
    groups = grupos_columnas(transformed_names)
    detail, grouped, hypotheses, baselines = [], [], [], []

    for part in ("aleatoria", "temporal"):
        split_col = "split_random" if part == "aleatoria" else "split_temporal"
        valid_df = pdf.loc[pdf[split_col].eq("validacion")].reset_index(drop=True)
        y = valid_df["objetivo"].to_numpy(dtype=np.int8)
        X0 = pre.transform(valid_df[cols])
        X = X0.toarray() if hasattr(X0, "toarray") else np.asarray(X0)
        X = np.ascontiguousarray(X, dtype=np.float32)
        models = load_ensemble(pre, part)
        p0 = predict_ensemble(X, models)
        base = metricas(y, p0)
        baselines.append({"particion": part, "n_validacion": len(y), "tasa_observada": float(y.mean()),
                          "pd_media_predicha": float(p0.mean()), **{f"baseline_{k}": v for k, v in base.items()}})
        rng = np.random.default_rng(np.random.SeedSequence([SEED_REPEAT, 0 if part == "aleatoria" else 1]))
        for name, ix in groups.items():
            deltas = []
            for repeat in range(args.repeats):
                perm = rng.permutation(len(y))
                Xp = X.copy()
                Xp[:, ix] = X[perm][:, ix]
                pp = predict_ensemble(Xp, models)
                met = metricas(y, pp)
                row = {"particion": part, "variable": name, "repeat": repeat,
                       "seed_permutacion": SEED_REPEAT, "n_validacion": len(y),
                       **{f"baseline_{k}": v for k, v in base.items()},
                       **{f"permutada_{k}": v for k, v in met.items()},
                       "delta_logloss": met["logloss"] - base["logloss"],
                       "delta_brier": met["brier"] - base["brier"],
                       "delta_auc": met["auc"] - base["auc"]}
                detail.append(row)
                deltas.append(row["delta_logloss"])
            grouped.append({"particion": part, "variable": name, "delta_logloss": float(np.mean(deltas)),
                            "delta_logloss_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                            "delta_logloss_min": float(np.min(deltas)), "delta_logloss_max": float(np.max(deltas)),
                            "repeticiones": args.repeats})
        hypotheses.append(hypothesis_groups(pdf, part, cols, p0))
        print(f"Completada partición {part}: {len(y):,} filas de validación; 24 grupos × {args.repeats} permutaciones.")

    pd.DataFrame(detail).to_csv(out / "permutacion_detalle.csv", index=False)
    summary = pd.DataFrame(grouped)
    summary.to_csv(out / "importancia_agrupada_validacion.csv", index=False)
    pd.DataFrame(baselines).to_csv(out / "metricas_base_validacion.csv", index=False)
    pd.concat(hypotheses, ignore_index=True).to_csv(out / "hipotesis_por_grupos_validacion.csv", index=False)
    write_plot(summary, out / "importancia_agrupada_validacion.png")
    manifest = {
        "metodo": "permutación agrupada en validación; impacto principal = delta log-loss del promedio de checkpoints 0-29",
        "repeticiones": args.repeats, "semillas_modelo": list(SEEDS), "semilla_permutacion": SEED_REPEAT,
        "grupos": {k: [transformed_names[i] for i in v] for k, v in groups.items()},
        "particiones_independientes": True, "prueba_consultada_para_resultados": False,
        "parquet_completo_cargado": True,
        "limitaciones": ["Importancia predictiva, no causal; la correlación puede repartir u ocultar importancia.",
                         "La evaluación temporal conserva la limitación del preprocesador ajustado con entrenamiento aleatorio.",
                         "Las columnas one-hot se permutan juntas y los indicadores con su variable base."],
        "tiempo_s": round(time.perf_counter() - t0, 3), "cpu_threads": args.num_threads,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "hashes": {"scripts/24_interpretabilidad_mlp_pd.py": sha256(Path(__file__)),
                   "src/mlp_pd.py": sha256(ROOT / "src/mlp_pd.py"),
                   "data/processed/pd.parquet": sha256(pdf_path), "models/preprocessor.joblib": sha256(pre_path),
                   **{f"models/pd_mlp_final_particionado/pd_mlp_{part}_final_seed_{seed}.pt":
                      sha256(ROOT / "models/pd_mlp_final_particionado" / f"pd_mlp_{part}_final_seed_{seed}.pt")
                      for part in ("aleatoria", "temporal") for seed in SEEDS},
                   "results/pd_mlp_final_particionado/pd_mlp_metricas_por_semilla.csv":
                      sha256(ROOT / "results/pd_mlp_final_particionado/pd_mlp_metricas_por_semilla.csv")},
    }
    (out / "manifiesto.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados: {out}\nDuración: {manifest['tiempo_s']:.1f} s")
    print(summary.sort_values(["particion", "delta_logloss"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
