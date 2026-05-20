#!/usr/bin/env python3
"""Build yearly ZIP-code adjacency matrices for Philadelphia ZIPs in the CSVs.

For each year, nodes are the ZIP codes appearing in that year's CSV. Edges are
created when two ZIP/ZCTA polygons share an exact boundary segment in the
downloaded Pennsylvania ZIP GeoJSON. Outputs are NumPy .npy files: one
adjacency matrix and one ZIP-code order array per year.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import os

import numpy as np


CSV_PATHS = [
    Path("../raw_data/zip_count/2022_homeless_yearly_by_zipcode.csv"),
    Path("../raw_data/zip_count/2023_homeless_yearly_by_zipcode.csv"),
    Path("../raw_data/zip_count/2024_homeless_yearly_by_zipcode.csv"),
    Path("../raw_data/zip_count/2025_homeless_yearly_by_zipcode.csv"),
]

GEOJSON_PATH = Path("pa_zipcodes.geojson")
YEARS = ["2022", "2023", "2024", "2025"]
OUTPUT_FOLDER = "data/"

def normalize_zip(value: str) -> str:
    value = value.strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(5)


def read_csv_data() -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    zips_by_year: dict[str, set[str]] = defaultdict(set)
    counts_by_year: dict[str, dict[str, int]] = defaultdict(dict)

    for path in CSV_PATHS:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                zipcode = normalize_zip(row["zipcode"])
                year = str(row["year"]).strip()
                if zipcode:
                    zips_by_year[year].add(zipcode)
                    counts_by_year[year][zipcode] = int(row["homeless_count"])

    return zips_by_year, counts_by_year


def polygon_rings(geometry: dict) -> list[list[list[float]]]:
    if geometry["type"] == "Polygon":
        return geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        rings = []
        for polygon in geometry["coordinates"]:
            rings.extend(polygon)
        return rings
    return []


def boundary_edges(features_by_zip: dict[str, dict]) -> set[tuple[str, str]]:
    segment_to_zips: dict[tuple[tuple[float, float], tuple[float, float]], set[str]] = defaultdict(set)

    for zipcode, feature in features_by_zip.items():
        for ring in polygon_rings(feature["geometry"]):
            for start, end in zip(ring, ring[1:]):
                p1 = tuple(round(coord, 6) for coord in start)
                p2 = tuple(round(coord, 6) for coord in end)
                if p1 == p2:
                    continue
                segment_to_zips[tuple(sorted((p1, p2)))].add(zipcode)

    edges: set[tuple[str, str]] = set()
    for zips in segment_to_zips.values():
        if len(zips) < 2:
            continue
        ordered = sorted(zips)
        for i, source in enumerate(ordered):
            for target in ordered[i + 1 :]:
                edges.add((source, target))

    return edges


def load_geojson_features() -> dict[str, dict]:
    with GEOJSON_PATH.open() as f:
        geojson = json.load(f)

    return {feature["properties"]["ZCTA5CE10"]: feature for feature in geojson["features"]}


def write_year_graph(
    year: str,
    csv_zips: set[str],
    homeless_counts: dict[str, int],
    all_features_by_zip: dict[str, dict],
) -> None:
    features_by_zip = {zipcode: all_features_by_zip[zipcode] for zipcode in csv_zips if zipcode in all_features_by_zip}

    edges = sorted(boundary_edges(features_by_zip))
    adjacency = {zipcode: [] for zipcode in sorted(csv_zips)}
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)

    for zipcode in adjacency:
        adjacency[zipcode] = sorted(adjacency[zipcode])

    graph_out = Path(f"{OUTPUT_FOLDER}/philly_zip_graph_{year}.npz")

    zipcodes = np.array(sorted(csv_zips), dtype=str)
    zipcode_to_index = {zipcode: index for index, zipcode in enumerate(zipcodes)}
    adjacency_matrix = np.zeros((len(zipcodes), len(zipcodes)), dtype=np.uint8)
    homeless_count = np.array([homeless_counts[zipcode] for zipcode in zipcodes], dtype=np.int64)
    for source, target in edges:
        source_index = zipcode_to_index[source]
        target_index = zipcode_to_index[target]
        adjacency_matrix[source_index, target_index] = 1
        adjacency_matrix[target_index, source_index] = 1

    np.savez_compressed(
        graph_out,
        adjacency=adjacency_matrix,
        zipcodes=zipcodes,
        homeless_count=homeless_count,
    )

    missing = sorted(csv_zips - set(features_by_zip))
    missing_text = f"; no boundary: {', '.join(missing)}" if missing else ""
    print(
        f"{year}: wrote {graph_out} "
        f"({len(csv_zips)} nodes, {len(edges)} edges{missing_text})"
    )


def main() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    zips_by_year, counts_by_year = read_csv_data()
    all_features_by_zip = load_geojson_features()

    for year in YEARS:
        write_year_graph(year, zips_by_year[year], counts_by_year[year], all_features_by_zip)


if __name__ == "__main__":
    main()
