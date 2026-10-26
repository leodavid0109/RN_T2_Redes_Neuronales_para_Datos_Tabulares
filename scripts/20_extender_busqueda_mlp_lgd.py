"""Extensión validation-only de un candidato LGD que alcanzó el máximo de épocas."""
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT, MODELS = ROOT / "results/lgd_mlp_search_extension_v2", ROOT / "models/lgd_mlp_search_extension_v2"
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
    cfgpath = ROOT / "configs/mlp_lgd_search_extension.yaml"
    cfg = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
    full_config = yaml.safe_load((ROOT / "configs/mlp_lgd.yaml").read_text(encoding="utf-8"))
    candidate = next(c for c in full_config["search"]["candidates"] if c["id"] == cfg["candidate_id"])
    torch.set_num_threads(cfg["num_threads"])
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    data = pd.read_parquet(ROOT / "data/processed/lgd.parquet")
    data = data.loc[data.castigado.astype(bool)].copy().reset_index(drop=True)
    pre = joblib.load(ROOT / "models/preprocessor.joblib")
    cols = list(pre.feature_names_in_)
    x = np.asarray(pre.transform(data[cols]), dtype=np.float32)
    y = data.lgd.to_numpy(dtype=np.float32)
    labels = data.split_random.to_numpy()
    ti, vi = np.flatnonzero(labels == "entrenamiento"), np.flatnonzero(labels == "validacion")
    ridge_pred = pd.read_parquet(ROOT / "results/lgd_lineal_pred.parquet")
    for subset, ix in (("entrenamiento", ti), ("validacion", vi)):
        expected = set(ridge_pred.loc[(ridge_pred.particion == "aleatoria") & (ridge_pred.conjunto == subset), "id"])
        if expected != set(data.iloc[ix].id): raise ValueError(f"IDs {subset} distintos de Ridge")
    inputs = ["data/processed/lgd.parquet", "models/preprocessor.joblib", "results/lgd_lineal.csv",
              "results/lgd_lineal_pred.parquet", "configs/mlp_lgd.yaml", "configs/mlp_lgd_search_extension.yaml",
              "src/mlp_lgd.py", "scripts/19_buscar_mlp_lgd.py", "scripts/20_extender_busqueda_mlp_lgd.py",
              "docs/mlp_lgd_busqueda.md"]
    hashes = {p: digest(ROOT / p) for p in inputs}
    base_metrics = pd.read_csv(ROOT / "results/lgd_lineal.csv")
    rows, histories, preds = [], {}, []
    t0 = time.perf_counter(); train_time = 0.0
    for seed in cfg["seeds"]:
        run = entrenar_una_corrida(x[ti], y[ti], x[vi], y[vi], candidate, seed=seed,
            batch_size=cfg["batch_size"], max_epochs=cfg["max_epochs"], patience=cfg["patience"],
            min_delta=cfg["min_delta"], num_threads=cfg["num_threads"],
            output_initialization=candidate["output_initialization"])
        if not np.isclose(run["best_validation_rmse"], min(h["valid_rmse"] for h in run["history"]), rtol=0, atol=1e-12):
            raise AssertionError("El checkpoint no es el mínimo RMSE de validación")
        train_time += run["training_seconds"]
        pred = predecir_lgd(run["model"], x[vi], cfg["batch_size"])
        obs = y[vi].astype(float)
        key = f"aleatoria/{candidate['id']}/seed_{seed}"; histories[key] = run["history"]
        rows.append({"configuracion":candidate["id"],"particion":"aleatoria","semilla":seed,
            "inicializacion":candidate["output_initialization"],"epoca_mejor":run["best_epoch"],
            "epocas_ejecutadas":run["epochs_run"],"motivo_parada":run["stop_reason"],
            "parametros_entrenables":run["parameters"],"tiempo_entrenamiento_s":run["training_seconds"],
            "media_predicha":float(pred.mean()),"media_observada":float(obs.mean()),**metricas_lgd(obs,pred)})
        preds.append(pd.DataFrame({"id":data.iloc[vi].id.to_numpy(),"semilla":seed,
            "lgd_observada":obs,"lgd_predicha":pred}))
        MODELS.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":run["state_dict"],"input_dim":int(x.shape[1]),"input_columns":cols,
            "config":candidate,"seed":seed,"partition":"aleatoria","best_epoch":run["best_epoch"],
            "epochs_run":run["epochs_run"],"parameters":run["parameters"],"hashes_input":hashes},
            MODELS/f"{candidate['id']}_seed_{seed}.pt")
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/"metricas_validacion.csv",index=False)
    pd.concat(preds,ignore_index=True).to_parquet(OUT/"predicciones_validacion.parquet",index=False)
    (OUT/"trayectorias_por_epoca.json").write_text(json.dumps(histories,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,5))
    for key,h in histories.items(): ax.plot([v["epoch"] for v in h],[v["valid_rmse"] for v in h],label=key)
    for name in ("ridge","media_train"):
        r=base_metrics.loc[(base_metrics.particion=="aleatoria")&(base_metrics.conjunto=="validacion")&(base_metrics.modelo==name)].iloc[0]
        ax.axhline(r.rmse,ls="--" if name=="ridge" else ":",label=f"{name} validación")
    ax.set(xlabel="Época",ylabel="RMSE validación",title="Extensión del contraste de inicialización")
    ax.legend();fig.tight_layout();fig.savefig(OUT/"curvas_extension_validacion.png",dpi=180);plt.close(fig)
    summary={"status":"validation_only_extension_no_final", "metrics_test_calculated":False,
        "reason":"Las dos semillas base con inicialización PyTorch alcanzaron la época máxima 100; repetidas desde inicio con máximo 160.",
        "config":cfg,"candidate":candidate,"results":rows,"training_seconds":train_time,
        "wall_seconds":time.perf_counter()-t0,"input_hashes":hashes,
        "versions":{"python":sys.version.split()[0],"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__}}
    (OUT/"resumen_extension.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"runs":len(rows),"results":str(OUT),"training_seconds":train_time,
        "wall_seconds":summary["wall_seconds"],"epochs":[r["epocas_ejecutadas"] for r in rows],
        "best_epochs":[r["epoca_mejor"] for r in rows]},indent=2))

if __name__=="__main__": main()
