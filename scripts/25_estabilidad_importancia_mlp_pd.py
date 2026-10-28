"""Diagnóstico de estabilidad entre semillas para importancia MLP de PD.

Usa cinco semillas equiespaciadas (0, 7, 14, 21, 29), fijadas sin mirar métricas,
y una permutación predefinida por grupo/partición. Solo usa validación. Complementa
la importancia principal del ensamble; no sustituye la estimación con 30 modelos.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pd_interp", ROOT / "scripts/24_interpretabilidad_mlp_pd.py")
interp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(interp)
MODEL_SEEDS = (0, 7, 14, 21, 29)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/pd_mlp_interpretabilidad_v2")
    ap.add_argument("--num-threads", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    detail_path = out / "estabilidad_por_semilla.csv"
    summary_path = out / "estabilidad_resumen.csv"
    if detail_path.exists() or summary_path.exists():
        raise FileExistsError("No sobrescribo el diagnóstico de estabilidad; use otro directorio.")
    torch.set_num_threads(args.num_threads)
    pdf = pd.read_parquet(ROOT / "data/processed/pd.parquet")
    pre = joblib.load(ROOT / "models/preprocessor.joblib")
    xcols = list(pre.feature_names_in_)
    transformed = list(pre.get_feature_names_out())
    groups = interp.grupos_columnas(transformed)
    rows = []
    for part_idx, part in enumerate(("aleatoria", "temporal")):
        split = "split_random" if part == "aleatoria" else "split_temporal"
        valid = pdf.loc[pdf[split].eq("validacion")].reset_index(drop=True)
        y = valid.objetivo.to_numpy(dtype=np.int8)
        xt = pre.transform(valid[xcols])
        X = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
        X = np.ascontiguousarray(X, dtype=np.float32)
        for seed in MODEL_SEEDS:
            path = ROOT / "models/pd_mlp_final_particionado" / f"pd_mlp_{part}_final_seed_{seed}.pt"
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = interp.MLPBinario(ck["input_dim"], ck["model_config"])
            model.load_state_dict(ck["state_dict"])
            p0 = interp.probabilidades_clase_1(model, X)
            base = interp.metricas(y, p0)
            for group_idx, (name, ix) in enumerate(groups.items()):
                rng = np.random.default_rng(np.random.SeedSequence([interp.SEED_REPEAT, part_idx, group_idx]))
                xp = X.copy()
                xp[:, ix] = X[rng.permutation(len(y))][:, ix]
                pp = interp.probabilidades_clase_1(model, xp)
                pert = interp.metricas(y, pp)
                rows.append({"particion": part, "semilla_modelo": seed, "variable": name,
                             "delta_logloss": pert["logloss"] - base["logloss"],
                             "delta_brier": pert["brier"] - base["brier"],
                             "delta_auc": pert["auc"] - base["auc"],
                             "n_validacion": len(y)})
        print(f"Estabilidad calculada en {part}: semillas {MODEL_SEEDS}.")
    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False)
    summary = detail.groupby(["particion", "variable"], as_index=False).agg(
        delta_logloss_media=("delta_logloss", "mean"),
        delta_logloss_sd_entre_semillas=("delta_logloss", "std"),
        delta_logloss_min=("delta_logloss", "min"),
        delta_logloss_max=("delta_logloss", "max"),
    )
    summary.to_csv(summary_path, index=False)
    print(summary.sort_values(["particion", "delta_logloss_media"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
