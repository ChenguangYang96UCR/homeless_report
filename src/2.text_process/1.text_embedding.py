from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

YEAR = 2025

CSV_PATH = Path(
    f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/raw_data/{YEAR}/{YEAR}_filtered_cases_fc_with_zip_lat_lon.csv"
)

OUT_DIR = Path(
    f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/text_embedding/text_embedding_cluster_{YEAR}"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
N_CLUSTERS = 5

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


TEXT_COLUMNS = [
    "subject",
    "status",
    "status_notes",
    "service_name",
    "service_code",
    "agency_responsible",
    "service_notice",
    "address",
]


def clean_zipcode(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if not value:
        return None

    return value.zfill(5)


def make_week(value):
    dt = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(dt):
        return None

    iso = dt.isocalendar()

    return f"week_{iso.year}-W{int(iso.week):02d}"


def build_text(row):
    parts = []

    for col in TEXT_COLUMNS:
        value = row.get(col)

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            parts.append(f"{col}: {value}")

    return " | ".join(parts)


def load_and_prepare_data():
    df = pd.read_csv(CSV_PATH)

    df["zipcode"] = df["zipcode"].apply(clean_zipcode)
    df["week"] = df["requested_datetime"].apply(make_week)
    df["text"] = df.apply(build_text, axis=1)

    df = df.dropna(subset=["zipcode", "week"])
    df = df[df["text"].str.len() > 0]

    df = df.sort_values(
        ["week", "zipcode", "objectid"],
        key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else x)
    ).reset_index(drop=True)

    return df


def embed_texts(df):
    model = SentenceTransformer(MODEL_NAME)

    texts = df["text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    metadata = df[
        [
            "objectid",
            "service_request_id",
            "week",
            "zipcode",
            "requested_datetime",
            "text",
        ]
    ].copy()

    return metadata, embeddings


def build_week_embeddings(metadata, text_embeddings):
    emb_dim = text_embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]

    emb_df = pd.DataFrame(text_embeddings, columns=emb_cols)

    case_level = pd.concat(
        [metadata.reset_index(drop=True), emb_df],
        axis=1
    )

    week_zip = (
        case_level
        .groupby(["week", "zipcode"], as_index=False)[emb_cols]
        .mean()
    )

    existing_weeks = sorted(metadata["week"].unique())
    rows = []

    for week in existing_weeks:
        week_df = week_zip[week_zip["week"] == week].set_index("zipcode")
        week_embedding = []
        has_text = []

        for zipcode in PHILLY_ZIPCODES:
            if zipcode in week_df.index:
                vec = week_df.loc[zipcode, emb_cols].to_numpy(dtype=float)
                has_text.append(1)
            else:
                vec = np.zeros(emb_dim, dtype=float)
                has_text.append(0)

            week_embedding.append(vec.tolist())

        rows.append({
            "week": week,
            "embedding": json.dumps(week_embedding),
            "has_text": json.dumps(has_text),
        })

    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def cluster_by_week(week_embeddings):
    rows = []

    for _, row in week_embeddings.iterrows():
        week = row["week"]
        embedding = json.loads(row["embedding"])
        has_text = json.loads(row["has_text"])

        X_all = np.array(embedding, dtype=float)
        has_text_arr = np.array(has_text, dtype=int) == 1
        n_samples = int(has_text_arr.sum())

        clusters = [-1] * len(PHILLY_ZIPCODES)

        if n_samples >= 2:
            k = min(N_CLUSTERS, n_samples)
            X = X_all[has_text_arr]

            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
            )

            labels = kmeans.fit_predict(X)
            label_index = 0

            for i, has_value in enumerate(has_text_arr):
                if has_value:
                    clusters[i] = int(labels[label_index])
                    label_index += 1

        elif n_samples == 1:
            only_index = int(np.where(has_text_arr)[0][0])
            clusters[only_index] = 0

        rows.append({
            "week": week,
            "clusters": json.dumps(clusters),
            "has_text": json.dumps(has_text),
        })

    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)

def main():
    print("Loading data...")
    df = load_and_prepare_data()
    print(f"Rows with valid text: {len(df)}")

    if df.empty:
        print("No valid text rows found.")
        return

    print("Embedding text...")
    metadata, text_embeddings = embed_texts(df)

    np.save(
        OUT_DIR / f"case_text_embeddings_{YEAR}.npy",
        text_embeddings
    )

    metadata.to_csv(
        OUT_DIR / f"case_text_embeddings_metadata_{YEAR}.csv",
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
        text_embeddings
    )

    week_embeddings[["week", "embedding"]].to_csv(
        OUT_DIR / f"weekly_text_embeddings_{YEAR}.csv",
        index=False
    )

    week_embeddings.to_csv(
        OUT_DIR / f"weekly_text_embeddings_with_has_text_{YEAR}.csv",
        index=False
    )

    print("Clustering by week...")
    clustered = cluster_by_week(week_embeddings)

    clustered.to_csv(
        OUT_DIR / f"weekly_text_clusters_{YEAR}.csv",
        index=False
    )

    print("Done.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()