from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from sklearn.cluster import KMeans
from tqdm import tqdm

YEAR = 2025

IMAGE_ROOT = Path(f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/satellite_images_esri/{YEAR}")
OUT_DIR = Path(f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/image_embedding/image_embedding_cluster_{YEAR}")
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


def build_week_embeddings(metadata, image_embeddings):
    emb_dim = image_embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]

    emb_df = pd.DataFrame(image_embeddings, columns=emb_cols)

    image_level = pd.concat(
        [metadata.reset_index(drop=True), emb_df],
        axis=1
    )

    week_zip = (
        image_level
        .groupby(["week", "zipcode"], as_index=False)[emb_cols]
        .mean()
    )

    existing_weeks = sorted(metadata["week"].unique())
    rows = []

    for week in existing_weeks:
        week_df = week_zip[week_zip["week"] == week].set_index("zipcode")
        week_embedding = []
        has_image = []

        for zipcode in PHILLY_ZIPCODES:
            if zipcode in week_df.index:
                vec = week_df.loc[zipcode, emb_cols].to_numpy(dtype=float)
                has_image.append(1)
            else:
                vec = np.zeros(emb_dim, dtype=float)
                has_image.append(0)

            week_embedding.append(vec.tolist())

        rows.append({
            "week": week,
            "embedding": json.dumps(week_embedding),
            "has_image": json.dumps(has_image),
        })

    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def cluster_by_week(week_embeddings):
    rows = []

    for _, row in week_embeddings.iterrows():
        week = row["week"]
        embedding = json.loads(row["embedding"])
        has_image = json.loads(row["has_image"])

        X_all = np.array(embedding, dtype=float)
        has_image_arr = np.array(has_image, dtype=int) == 1
        n_samples = int(has_image_arr.sum())

        clusters = [-1] * len(PHILLY_ZIPCODES)

        if n_samples >= 2:
            k = min(N_CLUSTERS, n_samples)
            X = X_all[has_image_arr]

            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
            )

            labels = kmeans.fit_predict(X)
            label_index = 0

            for i, has_value in enumerate(has_image_arr):
                if has_value:
                    clusters[i] = int(labels[label_index])
                    label_index += 1

        elif n_samples == 1:
            only_index = int(np.where(has_image_arr)[0][0])
            clusters[only_index] = 0

        rows.append({
            "week": week,
            "clusters": json.dumps(clusters),
            "has_image": json.dumps(has_image),
        })

    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)

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
        OUT_DIR / f"image_embeddings_{YEAR}.npy",
        image_embeddings
    )

    metadata.to_csv(
        OUT_DIR / f"image_embeddings_metadata_{YEAR}.csv",
        index=False
    )

    zipcode_order = pd.DataFrame({
        "zipcode_order": range(len(PHILLY_ZIPCODES)),
        "zipcode": PHILLY_ZIPCODES,
    })

    zipcode_order.to_csv(
        OUT_DIR / "philadelphia_zipcode_order.csv",
        index=False
    )

    print("Building weekly embeddings...")
    week_embeddings = build_week_embeddings(
        metadata,
        image_embeddings
    )

    week_embeddings[["week", "embedding"]].to_csv(
        OUT_DIR / f"weekly_image_embeddings_{YEAR}.csv",
        index=False
    )

    week_embeddings.to_csv(
        OUT_DIR / f"weekly_image_embeddings_with_has_image_{YEAR}.csv",
        index=False
    )

    print("Clustering by week...")
    clustered = cluster_by_week(week_embeddings)

    clustered.to_csv(
        OUT_DIR / f"weekly_image_clusters_{YEAR}.csv",
        index=False
    )

    print("Done.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()