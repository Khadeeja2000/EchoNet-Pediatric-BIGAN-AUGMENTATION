import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml


COLUMNS = [
    "view",
    "file_name",
    "file_path",
    "ef",
    "sex",
    "age",
    "weight",
    "height",
    "split",
    "age_bin",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def bin_age(age: float, age_bins: List[dict]) -> str:
    for b in age_bins:
        if b["min"] <= age <= b["max"]:
            return b["name"]
    return "other"


def read_filelist(view_dir: Path, view_name: str) -> pd.DataFrame:
    filelist_path = view_dir / "FileList.csv"
    if not filelist_path.exists():
        raise FileNotFoundError(f"Missing FileList.csv in {view_dir}")
    df = pd.read_csv(filelist_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"filename", "ef", "sex", "age", "weight", "height", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {filelist_path}: {missing}")
    df["view"] = view_name
    df["file_name"] = df["filename"]
    df["file_path"] = df["file_name"].apply(lambda x: str(view_dir / "Videos" / x))
    return df


def stratified_sample(
    df: pd.DataFrame,
    total_samples: int,
    balance_by: Tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    random.seed(seed)
    groups = df.groupby(list(balance_by), dropna=False)
    sizes = groups.size()
    proportions = sizes / sizes.sum()
    alloc = (proportions * total_samples).round().astype(int)
    alloc = alloc.mask(alloc == 0, 1)
    diff = alloc.sum() - total_samples
    if diff > 0:
        for idx in alloc.sort_values(ascending=False).index:
            if diff == 0:
                break
            if alloc.loc[idx] > 1:
                alloc.loc[idx] -= 1
                diff -= 1
    elif diff < 0:
        for idx in alloc.sort_values(ascending=False).index:
            if diff == 0:
                break
            alloc.loc[idx] += 1
            diff += 1

    samples = []
    for key, group in groups:
        n = alloc.get(key, 0)
        if n <= 0:
            continue
        take = min(n, len(group))
        samples.append(group.sample(n=take, random_state=seed))
    if not samples:
        return df.head(0)
    return pd.concat(samples, ignore_index=True)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_root = Path(cfg["paths"]["dataset_root"]).resolve()
    output_dir = Path(cfg["paths"]["output_dir"]).resolve()
    manifest_filename = cfg["paths"]["manifest_filename"]
    total_samples = int(cfg["subset"]["total_samples"]) if cfg["subset"].get("total_samples") else 500
    views: List[str] = list(cfg["subset"]["views"]) if cfg["subset"].get("views") else ["A4C", "PSAX"]
    balance_by = tuple(cfg["subset"]["balance_by"]) if cfg["subset"].get("balance_by") else ("sex", "age_bin")
    age_bins = list(cfg["subset"]["age_bins"]) if cfg["subset"].get("age_bins") else []
    seed = int(cfg.get("random_seed", 42))

    frames = []
    for view in views:
        view_dir = dataset_root / view
        df = read_filelist(view_dir, view)
        df["ef"] = pd.to_numeric(df["ef"], errors="coerce")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df["height"] = pd.to_numeric(df["height"], errors="coerce")
        df["age_bin"] = df["age"].apply(lambda a: bin_age(float(a), age_bins) if pd.notnull(a) else "other")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["file_path"].apply(lambda p: os.path.exists(p))]

    sampled = stratified_sample(
        all_df,
        total_samples=total_samples,
        balance_by=balance_by,
        seed=seed,
    )

    out_df = sampled[[
        "view",
        "file_name",
        "file_path",
        "ef",
        "sex",
        "age",
        "weight",
        "height",
        "split",
        "age_bin",
    ]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_filename
    out_df.to_csv(manifest_path, index=False)
    print(f"Wrote manifest with {len(out_df)} rows to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EchoNet Pediatric subset and write manifest")
    parser.add_argument("--config", type=str, default="preprocessing/config.yaml")
    args = parser.parse_args()
    main(args.config)
