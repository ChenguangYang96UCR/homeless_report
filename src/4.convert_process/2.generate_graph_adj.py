import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

GEOJSON = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/src/pa_zipcodes.geojson"
OUTPUT_DIR = Path("outputs/philly_2022_2025_gla_format")
ADJ_OUTPUT = OUTPUT_DIR / "philly_rn_adj.npy"
COUNT_DIR = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/raw_data/zip_count"
ZIP_ORDER = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/text_embedding/text_embedding_cluster_2022/philadelphia_zipcode_order.csv"
FULL_NPZ = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/src/4.convert_process/outputs/philly_2022_2025_gla_format/philly_2022_2025_full.npz"

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

    values = []
    for value in raw_values:
        value = str(value).strip()
        if not value or value.lower() == "nan":
            continue
        value = value.split(".")[0]
        if len(value) == 5 and value.isdigit():
            values.append(value)
    return values


def polygon_rings(geometry):
    if geometry["type"] == "Polygon":
        return geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        rings = []
        for polygon in geometry["coordinates"]:
            rings.extend(polygon)
        return rings
    return []


def boundary_edges(features_by_zip):
    segment_to_zips = defaultdict(set)
    for zipcode, feature in features_by_zip.items():
        for ring in polygon_rings(feature["geometry"]):
            for start, end in zip(ring, ring[1:]):
                p1 = tuple(round(coord, 6) for coord in start)
                p2 = tuple(round(coord, 6) for coord in end)
                if p1 == p2:
                    continue
                segment_to_zips[tuple(sorted((p1, p2)))].add(zipcode)

    edges = set()
    for zips in segment_to_zips.values():
        ordered = sorted(zips)
        for i, source in enumerate(ordered):
            for target in ordered[i + 1 :]:
                edges.add((source, target))
    return edges


def normalize_zip(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(5)


def read_weekly_counts(count_dir, years, zipcodes, week_starts):
    zip_set = set(zipcodes)
    frames = []
    zip_set = set(zipcodes)
    for year in years:
        path = Path(count_dir) / f"{year}_homeless_weekly_by_zipcode.csv"
        df = pd.read_csv(path)
        df["zip_norm"] = df["zipcode"].map(normalize_zip)
        df["week_start"] = df["week"].astype(str).str.split("/", expand=True)[0]
        frames.append(df[["week_start", "zip_norm", "homeless_count"]])
        dropped = sorted(set(df["zip_norm"]) - zip_set)
        if dropped:
            print(f"{year}: count ZIPs not in graph order, dropped: {dropped}")

    all_counts = pd.concat(frames, ignore_index=True)
    grouped = all_counts.groupby(["week_start", "zip_norm"], as_index=False)["homeless_count"].sum()
    pivot = (
        grouped.pivot(index="week_start", columns="zip_norm", values="homeless_count")
        .reindex(index=week_starts, columns=zipcodes, fill_value=0)
        .fillna(0)
        .astype(np.int64)
    )
    return pivot.to_numpy(dtype=np.int64).T


def load_week_starts(npz_path):
    if npz_path is None:
        return None
    data = np.load(npz_path, allow_pickle=True)
    return pd.to_datetime(data["start_time"]).strftime("%Y-%m-%d").tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-order", default=ZIP_ORDER)
    parser.add_argument("--geojson", default=GEOJSON)
    parser.add_argument("--output", default=str(ADJ_OUTPUT))
    parser.add_argument("--graph-output", default=str(OUTPUT_DIR / "philly_zip_graph_2022_2025.npz"))
    parser.add_argument("--nodes-output", default=str(OUTPUT_DIR / "philly_zip_graph_nodes_2022_2025.csv"))
    parser.add_argument("--node-weekly-output", default=str(OUTPUT_DIR / "philly_zip_graph_node_weekly_counts_2022_2025.csv"))
    parser.add_argument("--edges-output", default=str(OUTPUT_DIR / "philly_zip_graph_edges_2022_2025.csv"))
    parser.add_argument("--count-dir", default=COUNT_DIR)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--weeks-from-npz", default=FULL_NPZ)
    parser.add_argument("--self-loops", action="store_true")
    args = parser.parse_args()

    zipcodes = parse_zip_order(args.zip_order)
    with open(args.geojson) as f:
        geojson = json.load(f)
    all_features = {feature["properties"]["ZCTA5CE10"]: feature for feature in geojson["features"]}
    features_by_zip = {zipcode: all_features[zipcode] for zipcode in zipcodes if zipcode in all_features}

    zipcode_to_index = {zipcode: index for index, zipcode in enumerate(zipcodes)}
    adj = np.zeros((len(zipcodes), len(zipcodes)), dtype=np.float32)
    edges = sorted(
        (source, target)
        for source, target in boundary_edges(features_by_zip)
        if source in zipcode_to_index and target in zipcode_to_index
    )
    for source, target in edges:
        if source in zipcode_to_index and target in zipcode_to_index:
            i = zipcode_to_index[source]
            j = zipcode_to_index[target]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    if args.self_loops:
        np.fill_diagonal(adj, 1.0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, adj)

    week_starts = load_week_starts(args.weeks_from_npz)
    if week_starts is None:
        raise ValueError("--weeks-from-npz is required so weekly node properties align with model data.")
    weekly_counts = read_weekly_counts(args.count_dir, args.years, zipcodes, week_starts)
    total_count = weekly_counts.sum(axis=1)

    graph_output = Path(args.graph_output) if args.graph_output else output.with_name("philly_zip_graph_2022_2025.npz")
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        graph_output,
        adjacency=adj,
        zipcodes=np.array(zipcodes, dtype=str),
        years=np.array(args.years, dtype=np.int64),
        weeks=np.array(week_starts, dtype=str),
        weekly_counts=weekly_counts,
        total_count=total_count,
        edges=np.array(edges, dtype=str),
    )

    nodes_output = Path(args.nodes_output) if args.nodes_output else output.with_name("philly_zip_graph_nodes_2022_2025.csv")
    node_rows = []
    for idx, zipcode in enumerate(zipcodes):
        row = {"node_id": idx, "zipcode": zipcode, "weekly_counts": json.dumps(weekly_counts[idx].astype(int).tolist())}
        row["count_2022_2025_total"] = int(total_count[idx])
        node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(nodes_output, index=False)

    node_weekly_output = (
        Path(args.node_weekly_output)
        if args.node_weekly_output
        else output.with_name("philly_zip_graph_node_weekly_counts_2022_2025.csv")
    )
    weekly_rows = []
    for node_id, zipcode in enumerate(zipcodes):
        for week_idx, week_start in enumerate(week_starts):
            weekly_rows.append(
                {
                    "node_id": node_id,
                    "zipcode": zipcode,
                    "week_index": week_idx,
                    "week_start": week_start,
                    "homeless_count": int(weekly_counts[node_id, week_idx]),
                }
            )
    pd.DataFrame(weekly_rows).to_csv(node_weekly_output, index=False)

    edges_output = Path(args.edges_output) if args.edges_output else output.with_name("philly_zip_graph_edges_2022_2025.csv")
    edge_rows = []
    for source, target in edges:
        edge_rows.append(
            {
                "source_id": zipcode_to_index[source],
                "target_id": zipcode_to_index[target],
                "source_zipcode": source,
                "target_zipcode": target,
            }
        )
    pd.DataFrame(edge_rows).to_csv(edges_output, index=False)

    missing = sorted(set(zipcodes) - set(features_by_zip))
    print(f"wrote {output} shape={adj.shape} edges={len(edges)} missing_geojson={missing}")
    print(f"wrote {graph_output}")
    print(f"wrote {nodes_output}")
    print(f"wrote {node_weekly_output}")
    print(f"wrote {edges_output}")


if __name__ == "__main__":
    main()
