from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from sklearn.cluster import KMeans
from tqdm import tqdm


IMAGE_ROOT = Path("/Users/chenguangyang/Desktop/ucr_work/homeless_report/satellite_images_esri/2025")
OUT_DIR = Path("/Users/chenguangyang/Desktop/ucr_work/homeless_report/embedding_cluster_2025")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 5
BATCH_SIZE = 32

PHILLY_ZIPCODES = [
    "19102", "19103", "19104", "19106", "19107",
    "19111", "19112", "19114", "19115", "19116",
    "19118", "19119", "19120", "19121", "19122",
    "19123", "19124", "19125", "19126", "19127",
    "19128", "19129", "19130", "19131", "19132",
    "19133", "19134", "19135", "19136", "19137",
    "19138", "19139", "19140", "19141", "19142",
    "19143", "19144", "19145", "19146", "19147",
    "19148", "19149", "19150", "19151", "19152",
    "19153", "19154",
]

PHILLY_ZIPCODES = sorted(PHILLY_ZIPCODES, key=lambda x: int(x))


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    transform = weights.transforms()

    return model, transform, device


def parse_week_and_zip(image_path):
    week = None

    for part in image_path.parts:
        if re.match(r"week_\d{4}-W\d{2}$", part):
            week = part

    zipcode = image_path.parent.name

    return week, zipcode


def collect_images():
    image_paths = []

    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(IMAGE_ROOT.glob(f"week_*/*/{ext}"))

    rows = []

    for path in image_paths:
        week, zipcode = parse_week_and_zip(path)

        if week is None:
            continue

        rows.append({
            "week": week,
            "zipcode": str(zipcode),
            "image_path": str(path),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["zipcode"] = df["zipcode"].astype(str)
    df = df.sort_values(["week", "zipcode", "image_path"]).reset_index(drop=True)

    return df


def embed_images(df, model, transform, device):
    image_embeddings = []
    metadata_rows = []

    paths = df["image_path"].tolist()

    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="Embedding images"):
        batch_paths = paths[start:start + BATCH_SIZE]
        batch_tensors = []
        valid_rows = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                tensor = transform(image)
                batch_tensors.append(tensor)
                valid_rows.append(path)
            except Exception as e:
                print(f"skip bad image: {path}, error={e}")

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)

        with torch.inference_mode():
            emb = model(batch).detach().cpu().numpy()

        image_embeddings.append(emb)

        for path in valid_rows:
            row = df[df["image_path"] == path].iloc[0]
            metadata_rows.append({
                "week": row["week"],
                "zipcode": row["zipcode"],
                "image_path": path,
            })

    image_embeddings = np.vstack(image_embeddings)
    metadata = pd.DataFrame(metadata_rows)

    return metadata, image_embeddings


def sorted_embedding_columns(columns):
    return sorted(
        [c for c in columns if c.startswith("emb_")],
        key=lambda x: int(x.split("_")[1])
    )


def build_week_zip_embeddings(metadata, image_embeddings):
    emb_dim = image_embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]

    emb_df = pd.DataFrame(image_embeddings, columns=emb_cols)

    image_level = pd.concat(
        [metadata.reset_index(drop=True), emb_df],
        axis=1
    )

    zip_week = (
        image_level
        .groupby(["week", "zipcode"], as_index=False)[emb_cols]
        .mean()
    )

    existing_weeks = sorted(metadata["week"].unique())
    all_rows = []

    for week in existing_weeks:
        week_df = zip_week[zip_week["week"] == week].set_index("zipcode")

        for zipcode in PHILLY_ZIPCODES:
            if zipcode in week_df.index:
                vec = week_df.loc[zipcode, emb_cols].to_numpy(dtype=float)
                has_image = 1
            else:
                vec = np.zeros(emb_dim, dtype=float)
                has_image = 0

            row = {
                "week": week,
                "zipcode": zipcode,
                "has_image": has_image,
            }

            for i, value in enumerate(vec):
                row[f"emb_{i}"] = value

            all_rows.append(row)

    out = pd.DataFrame(all_rows)

    emb_cols = sorted_embedding_columns(out.columns)

    out = out[["week", "zipcode", "has_image"] + emb_cols]
    out = out.sort_values(
        ["week", "zipcode"],
        key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else x)
    ).reset_index(drop=True)

    return out


def cluster_by_week(week_zip_embeddings):
    emb_cols = sorted_embedding_columns(week_zip_embeddings.columns)
    clustered_rows = []

    for week, week_df in week_zip_embeddings.groupby("week"):
        week_df = week_df.copy()

        nonzero = week_df["has_image"] == 1
        n_samples = int(nonzero.sum())

        week_df["cluster"] = -1

        if n_samples >= 2:
            k = min(N_CLUSTERS, n_samples)
            X = week_df.loc[nonzero, emb_cols].to_numpy(dtype=float)

            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init="auto"
            )

            week_df.loc[nonzero, "cluster"] = kmeans.fit_predict(X)

        elif n_samples == 1:
            week_df.loc[nonzero, "cluster"] = 0

        clustered_rows.append(week_df)

    out = pd.concat(clustered_rows, ignore_index=True)

    emb_cols = sorted_embedding_columns(out.columns)

    out = out[["week", "zipcode", "has_image", "cluster"] + emb_cols]
    out = out.sort_values(
        ["week", "zipcode"],
        key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else x)
    ).reset_index(drop=True)

    return out


def main():
    print("Collecting images...")
    df = collect_images()
    print(f"Found images: {len(df)}")

    if df.empty:
        print("No images found.")
        return

    model, transform, device = load_model()
    print(f"Using device: {device}")

    metadata, image_embeddings = embed_images(df, model, transform, device)

    np.save(
        OUT_DIR / "image_embeddings_2025.npy",
        image_embeddings
    )

    metadata.to_csv(
        OUT_DIR / "image_embeddings_metadata_2025.csv",
        index=False
    )

    week_zip_embeddings = build_week_zip_embeddings(
        metadata,
        image_embeddings
    )

    week_zip_embeddings.to_csv(
        OUT_DIR / "weekly_zip_embeddings_2025.csv",
        index=False
    )

    clustered = cluster_by_week(week_zip_embeddings)

    clustered.to_csv(
        OUT_DIR / "weekly_zip_embeddings_clusters_2025.csv",
        index=False
    )

    cluster_only = clustered[["week", "zipcode", "has_image", "cluster"]]

    cluster_only.to_csv(
        OUT_DIR / "weekly_zip_clusters_2025.csv",
        index=False
    )

    print("Done.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()