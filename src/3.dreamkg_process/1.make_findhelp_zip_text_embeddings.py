import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FILES = [
    "/Users/chenguangyang/Desktop/work/DREAM-KG/TheWebConf/data/New_FindHelp_philadelphia_mental_health_2025_0707.csv",
    "/Users/chenguangyang/Desktop/work/DREAM-KG/TheWebConf/data/New_FindHelp_philadelphia_temporary_shelter_2025_0707.csv",
    "/Users/chenguangyang/Desktop/work/DREAM-KG/TheWebConf/data/New_philadelphia_emergency_food_2025_0707.csv",
]
ZIP_ORDER = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/text_embedding/text_embedding_cluster_2025/philadelphia_zipcode_order.csv"
BASE_GRAPH_NPZ = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/src/4.convert_process/outputs/philly_2022_2025_gla_format/philly_zip_graph_2022_2025.npz"
OUTPUT_DIR = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/src/3.dreamkg_process/outputs/findhelp_zip_graph"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_zip_order(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if lines and all("," not in line for line in lines) and lines[0].split(".")[0].isdigit():
        raw_values = lines
    else:
        zip_df = pd.read_csv(path, dtype=str)
        if "zipcode" in zip_df.columns:
            raw_values = zip_df["zipcode"].tolist()
        elif zip_df.shape[1] >= 2 and zip_df.iloc[:, 1].astype(str).str.fullmatch(r"\d{5}(?:\.0)?").all():
            raw_values = zip_df.iloc[:, 1].tolist()
        else:
            raw_values = zip_df.iloc[:, 0].tolist()
    return [str(v).strip().split(".")[0] for v in raw_values if len(str(v).strip().split(".")[0]) == 5]


def normalize_zip(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"\d{5}", text)
    return match.group(0) if match else None


def clean_value(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_findhelp_rows(paths):
    rows = []
    text_columns = [
        "Service_name",
        "Service_Name",
        "Main_Services",
        "Other_Services",
        "Serving",
        "Eligibility",
        "Description",
        "Languages",
        "Cost",
        "Coverage",
        "Service",
        "Service_type",
    ]
    for path in paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            zipcode = normalize_zip(row.get("Zipcode"))
            if zipcode is None:
                continue
            parts = [clean_value(row.get(col)) for col in text_columns if col in df.columns]
            text = " ".join(part for part in parts if part)
            rows.append(
                {
                    "zipcode": zipcode,
                    "source_file": Path(path).name,
                    "service_type": clean_value(row.get("Service", row.get("Service_type", ""))),
                    "text": text,
                }
            )
    return pd.DataFrame(rows)


def resize_embeddings(embeddings, target_dim):
    if target_dim is None or embeddings.shape[1] == target_dim:
        return embeddings
    out = np.zeros((embeddings.shape[0], target_dim), dtype=np.float32)
    keep = min(embeddings.shape[1], target_dim)
    out[:, :keep] = embeddings[:, :keep]
    return out


def encode_documents(documents, model_name, batch_size):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence_transformers is required to align with homeless text embeddings. "
            "Install it in the same environment, or run this script on the server where "
            "the homeless embeddings were generated."
        ) from exc
    model = SentenceTransformer(model_name)
    return model.encode(
        documents,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--zip-order", default=ZIP_ORDER)
    parser.add_argument("--base-graph-npz", default=BASE_GRAPH_NPZ)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resize-to", type=int, default=0, help="Optional output dimension. 0 keeps model dimension.")
    args = parser.parse_args()

    zipcodes = parse_zip_order(args.zip_order)
    zip_set = set(zipcodes)
    rows = read_findhelp_rows(args.files)
    dropped = sorted(set(rows["zipcode"]) - zip_set)
    if dropped:
        print(f"FindHelp ZIPs not in graph order, dropped: {dropped}")
    rows = rows[rows["zipcode"].isin(zip_set)].copy()

    grouped_text = rows.groupby("zipcode")["text"].apply(lambda values: " ".join(values)).to_dict()
    documents = [grouped_text.get(zipcode, "") for zipcode in zipcodes]
    embeddings_raw = encode_documents(documents, args.model_name, args.batch_size)
    resize_to = None if args.resize_to == 0 else args.resize_to
    embeddings = resize_embeddings(embeddings_raw, resize_to)

    service_counts = rows.groupby("zipcode").size().reindex(zipcodes, fill_value=0).to_numpy(dtype=np.int64)
    service_type_counts = (
        rows.pivot_table(index="zipcode", columns="service_type", values="text", aggfunc="count", fill_value=0)
        .reindex(index=zipcodes, fill_value=0)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_csv = output_dir / "philly_findhelp_zip_text_embeddings.csv"
    pd.DataFrame(
        {
            "list_id": list(range(len(zipcodes))),
            "zipcode": zipcodes,
            "service_count": service_counts,
            "text_embedding": [json.dumps(vec.astype(float).tolist()) for vec in embeddings],
        }
    ).to_csv(embedding_csv, index=False)

    rows_csv = output_dir / "philly_findhelp_services_used.csv"
    rows.to_csv(rows_csv, index=False)

    graph_path = Path(args.base_graph_npz)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Base graph file not found: {graph_path}. Pass --base-graph-npz with the path to "
            "the ZIP adjacency graph. It will only be read; it will not be modified."
        )
    base_graph = np.load(graph_path, allow_pickle=True)
    base_zipcodes = [str(z) for z in base_graph["zipcodes"].tolist()]
    if base_zipcodes != zipcodes:
        raise ValueError("ZIP order mismatch between --zip-order and --base-graph-npz.")
    adjacency = base_graph["adjacency"].astype(np.float32)
    edges = base_graph["edges"] if "edges" in base_graph.files else np.empty((0, 2), dtype=str)

    graph_output = output_dir / "findhelp_zip_graph.npz"
    np.savez_compressed(
        graph_output,
        adjacency=adjacency,
        zipcodes=np.array(zipcodes, dtype=str),
        findhelp_text_embeddings=embeddings,
        findhelp_service_count=service_counts,
        findhelp_service_type_names=service_type_counts.columns.to_numpy(dtype=str),
        findhelp_service_type_counts=service_type_counts.to_numpy(dtype=np.int64),
        edges=edges,
    )

    nodes_output = output_dir / "findhelp_zip_graph_nodes.csv"
    node_rows = []
    for idx, zipcode in enumerate(zipcodes):
        node_rows.append(
            {
                "node_id": idx,
                "zipcode": zipcode,
                "service_count": int(service_counts[idx]),
                "text_embedding": json.dumps(embeddings[idx].astype(float).tolist()),
            }
        )
    pd.DataFrame(node_rows).to_csv(nodes_output, index=False)

    edges_output = output_dir / "findhelp_zip_graph_edges.csv"
    edge_rows = []
    for edge in edges:
        source, target = str(edge[0]), str(edge[1])
        edge_rows.append(
            {
                "source_id": zipcodes.index(source),
                "target_id": zipcodes.index(target),
                "source_zipcode": source,
                "target_zipcode": target,
            }
        )
    pd.DataFrame(edge_rows).to_csv(edges_output, index=False)

    manifest = {
        "zip_count": len(zipcodes),
        "model_name": args.model_name,
        "raw_embedding_shape": list(embeddings_raw.shape),
        "embedding_shape": list(embeddings.shape),
        "service_rows_used": int(len(rows)),
        "zips_with_services": int((service_counts > 0).sum()),
        "dropped_zips": dropped,
        "source_files": [str(p) for p in args.files],
        "outputs": {
            "embedding_csv": str(embedding_csv),
            "rows_csv": str(rows_csv),
            "graph_npz": str(graph_output),
            "nodes_csv": str(nodes_output),
            "edges_csv": str(edges_output),
        },
        "method": "SentenceTransformer ZIP-level aggregated document embedding. Default model all-MiniLM-L6-v2 outputs 384 dimensions, matching homeless text embeddings.",
        "base_graph_npz_read_only": str(graph_path),
    }
    manifest_path = output_dir / "philly_findhelp_text_embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
