"""Extensión determinista, solo validación, del piloto de convergencia LGD."""
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT, MODELS = ROOT / "results/lgd_mlp_convergencia_extension_v2", ROOT / "models/lgd_mlp_convergencia_extension_v2"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv" / "matplotlib-config"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, numpy as np, pandas as pd, torch, yaml
sys.path.insert(0, str(ROOT))
from src.metricas_lgd import metricas_lgd
from src.mlp_lgd import entrenar_una_corrida, predecir_lgd


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    for p in (OUT, MODELS):
        if p.exists() and any(p.iterdir()): raise FileExistsError(f"No sobrescribo {p}")
    cfg_path = ROOT / "configs/mlp_lgd_convergencia_extension.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    torch.set_num_threads(cfg["num_threads"])
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    data = pd.read_parquet(ROOT / "data/processed/lgd.parquet")
    data = data.loc[data.castigado.astype(bool)].copy().reset_index(drop=True)
    pre = joblib.load(ROOT / "models/preprocessor.joblib")
    cols = list(pre.feature_names_in_)
    x = np.asarray(pre.transform(data[cols]), dtype=np.float32)
    y = data.lgd.to_numpy(dtype=np.float32)
    split_col = "split_random"
    labels = data[split_col].to_numpy()
    ti, vi = np.flatnonzero(labels == "entrenamiento"), np.flatnonzero(labels == "validacion")
    ridge_pred = pd.read_parquet(ROOT / "results/lgd_lineal_pred.parquet")
    for s, ix in (("entrenamiento", ti), ("validacion", vi)):
        expected = set(ridge_pred.loc[(ridge_pred.particion == cfg["partition"]) & (ridge_pred.conjunto == s), "id"])
        if expected != set(data.iloc[ix].id): raise ValueError(f"IDs {s} distintos de Ridge")
    inputs = ["data/processed/lgd.parquet", "models/preprocessor.joblib", "results/lgd_lineal.csv", "results/lgd_lineal_pred.parquet", "src/mlp_lgd.py", "scripts/18_extender_convergencia_lgd.py", "configs/mlp_lgd_convergencia_extension.yaml", "docs/mlp_lgd_convergencia.md"]
    hashes = {p: digest(ROOT / p) for p in inputs}
    histories, rows, pred_frames = {}, [], []
    t0 = time.perf_counter(); total_train = 0.0
    for seed in cfg["seeds"]:
        run = entrenar_una_corrida(x[ti], y[ti], x[vi], y[vi], cfg["model"], seed=seed,
            batch_size=cfg["batch_size"], max_epochs=cfg["max_epochs"], patience=cfg["patience"],
            min_delta=cfg["min_delta"], num_threads=cfg["num_threads"], output_initialization="pytorch")
        total_train += run["training_seconds"]
        key = f"aleatoria/pytorch/seed_{seed}"; histories[key] = run["history"]
        pred = predecir_lgd(run["model"], x[vi], cfg["batch_size"])
        obs = y[vi].astype(float)
        rows.append({"particion":"aleatoria", "inicializacion":"pytorch", "semilla":seed,
            "epoca_mejor":run["best_epoch"], "epocas_ejecutadas":run["epochs_run"], "motivo_parada":run["stop_reason"],
            "parametros_entrenables":run["parameters"], "tiempo_entrenamiento_s":run["training_seconds"],
            "media_predicha":float(pred.mean()), "media_observada":float(obs.mean()), **metricas_lgd(obs,pred)})
        pred_frames.append(pd.DataFrame({"id":data.iloc[vi].id.to_numpy(), "semilla":seed, "lgd_observada":obs, "lgd_predicha":pred}))
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict":run["state_dict"], "input_dim":int(x.shape[1]), "input_columns":cols,
            "config":cfg["model"], "seed":seed, "partition":"aleatoria", "best_epoch":run["best_epoch"],
            "epochs_run":run["epochs_run"], "parameters":run["parameters"], "hashes_input":hashes},
            MODELS / f"aleatoria_pytorch_seed_{seed}.pt")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "metricas_validacion.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(OUT / "predicciones_validacion.parquet", index=False)
    (OUT / "trayectorias_por_epoca.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8,5))
    for k,h in histories.items():
        ax.plot([v["epoch"] for v in h],[v["valid_rmse"] for v in h],label=k)
    ax.axhline(.09474611970124572,color="black",ls="--",label="Ridge validación")
    ax.axhline(.09543572284628865,color="gray",ls=":",label="Media train validación")
    ax.set(xlabel="Época",ylabel="RMSE validación",title="Extensión aleatoria: inicialización PyTorch")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "curvas_extension_validacion.png",dpi=180); plt.close(fig)
    summary={"estado":"extension_exploratoria_no_final_validation_only", "no_se_calcularon_metricas_prueba":True,
        "motivo":"Dos corridas del piloto agotaron 60 épocas; se repiten desde inicio con máximo 90 porque no se guardó estado Adam.",
        "configuracion":cfg,"resultados":rows,"tiempo_entrenamiento_s":total_train,"tiempo_pared_s":time.perf_counter()-t0,
        "budget_seconds":cfg["budget_seconds"],"hashes_entrada":hashes,
        "versiones":{"python":sys.version.split()[0],"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__}}
    (OUT / "resumen_extension.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"corridas":len(rows),"tiempo_entrenamiento_s":total_train,"tiempo_pared_s":summary["tiempo_pared_s"],"salida":str(OUT)},indent=2))

if __name__ == "__main__": main()
