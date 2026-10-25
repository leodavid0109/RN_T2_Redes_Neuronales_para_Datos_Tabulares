"""Construye la variable objetivo LGD y data/processed/lgd.parquet.

Uso:
    python src/lgd.py
    python src/lgd.py --raw data/raw/loan.csv --pd data/processed/pd.parquet
"""
import argparse
import pandas as pd

AUX = ["loan_status", "funded_amnt", "total_rec_prncp",
       "recoveries", "collection_recovery_fee"]


def build_lgd(raw_path, pd_path, out_path):
    pdf = pd.read_parquet(pd_path)
    raw = pd.read_csv(raw_path, usecols=["id"] + AUX, low_memory=False)

    # Mismo tipo de id para poder unir
    raw["id"] = pd.to_numeric(raw["id"], errors="coerce")
    pdf["id"] = pd.to_numeric(pdf["id"], errors="coerce")

    # Solo incumplidos (objetivo = 1); hereda las mismas particiones de PD
    bad = pdf[pdf["objetivo"] == 1]
    df = bad.merge(raw, on="id", how="left", validate="one_to_one")
    print(f"Incumplidos en pd.parquet: {len(bad):,} | "
          f"sin cruce con raw: {df['funded_amnt'].isna().sum():,}")
    df = df.dropna(subset=["funded_amnt"])

    # EAD = capital pendiente al incumplir
    df["ead"] = df["funded_amnt"] - df["total_rec_prncp"]
    n_bad_ead = (df["ead"] <= 0).sum()
    print(f"Excluidos por EAD <= 0: {n_bad_ead:,}")
    df = df[df["ead"] > 0].copy()

    # LGD = 1 - recuperacion neta / EAD, acotada a [0, 1]
    neto = df["recoveries"] - df["collection_recovery_fee"]
    df["lgd"] = (1 - neto / df["ead"]).clip(0, 1)

    CASTIGADOS = ["Charged Off",
                "Does not meet the credit policy. Status:Charged Off"]
    df["castigado"] = df["loan_status"].isin(CASTIGADOS)
    print(f"Castigados (muestra principal): {df['castigado'].sum():,}")

    # Diagnostico de la distribucion
    print("\nResumen LGD:")
    print(df["lgd"].describe())
    print(f"Masa en LGD == 1: {(df['lgd'] == 1).mean():.1%}")
    print(f"Masa en LGD == 0: {(df['lgd'] == 0).mean():.1%}")
    print("\nLGD por estado original:")
    print(df.groupby("loan_status")["lgd"].agg(["count", "mean"]))

    # Las columnas posteriores NO son caracteristicas del modelo.
    # Se conservan solo 'lgd', 'ead' y 'funded_amnt' (para perdida esperada).
    df = df.drop(columns=["total_rec_prncp", "recoveries",
                          "collection_recovery_fee"])
    df.to_parquet(out_path, index=False)
    print(f"\nGuardado: {out_path} {df.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/loan.csv")
    ap.add_argument("--pd", default="data/processed/pd.parquet")
    ap.add_argument("--out", default="data/processed/lgd.parquet")
    a = ap.parse_args()
    build_lgd(a.raw, a.pd, a.out)