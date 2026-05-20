import os
import re
import time
import math
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import RequestException

YEAR = 2025

CSV_PATH = f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/raw_data/{YEAR}/{YEAR}_filtered_cases_fc_with_zip_lat_lon.csv"
OUT_DIR = f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/satellite_images_esri/{YEAR}"
ZIP_GEOJSON_PATH = f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/src/pa_zipcodes.geojson"

URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"


def lonlat_to_web_mercator(lon, lat):
    origin_shift = 20037508.342789244
    x = lon * origin_shift / 180.0
    lat = max(min(lat, 89.5), -89.5)
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * origin_shift / 180.0
    return x, y


def bbox_around_point(lon, lat, half_size_m=80):
    x, y = lonlat_to_web_mercator(lon, lat)
    return f"{x-half_size_m},{y-half_size_m},{x+half_size_m},{y+half_size_m}"


def safe_text(value):
    value = str(value).strip()
    value = value.replace("+00", "")
    value = value.replace(" ", "_").replace(":", "-")
    return re.sub(r"[^0-9A-Za-z_.-]", "", value)


def clean_zipcode(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(5)


def week_folder(timestamp):
    # ISO week: Monday-Sunday. Example: 2020-W01.
    iso = timestamp.isocalendar()
    return f"week_{iso.year}-W{iso.week:02d}"


def make_image_name(row):
    dt = safe_text(row["requested_datetime"])
    lat = f"{float(row['lat']):.8f}"
    lon = f"{float(row['lon']):.8f}"
    objectid = safe_text(row["objectid"])
    return f"{dt}_{lat}_{lon}_{objectid}.png"


def geometry_polygons(geometry):
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    return []


def geometry_rings(geometry):
    rings = []
    for polygon in geometry_polygons(geometry):
        rings.extend(polygon)
    return rings


def point_in_ring(lon, lat, ring):
    inside = False
    j = len(ring) - 1
    for i, point in enumerate(ring):
        xi, yi = point[0], point[1]
        xj, yj = ring[j][0], ring[j][1]
        intersects = (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def point_in_geometry(lon, lat, geometry):
    for polygon in geometry_polygons(geometry):
        exterior = polygon[0]
        holes = polygon[1:]
        if point_in_ring(lon, lat, exterior) and not any(point_in_ring(lon, lat, hole) for hole in holes):
            return True
    return False


def load_zip_lookup(path):
    import json

    with open(path) as f:
        geojson = json.load(f)

    lookup = []
    for feature in geojson["features"]:
        zipcode = feature["properties"]["ZCTA5CE10"]
        rings = geometry_rings(feature["geometry"])
        if not rings:
            continue
        coords = [point for ring in rings for point in ring]
        lons = [point[0] for point in coords]
        lats = [point[1] for point in coords]
        lookup.append(
            {
                "zipcode": zipcode,
                "geometry": feature["geometry"],
                "bbox": (min(lons), min(lats), max(lons), max(lats)),
            }
        )
    return lookup


def infer_zipcode_from_point(lon, lat, zip_lookup):
    for item in zip_lookup:
        min_lon, min_lat, max_lon, max_lat = item["bbox"]
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            continue
        if point_in_geometry(lon, lat, item["geometry"]):
            return item["zipcode"]
    return "unknown_zipcode"


def download_image_with_retry(url, params, out_path, objectid=None, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(10, 60),  # 10seconds
                stream=True
            )

            content_type = response.headers.get("content-type", "")

            if response.status_code == 200 and content_type.startswith("image"):
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)

                tmp_path.replace(out_path)
                print("saved", out_path)
                return True

            print(
                "failed",
                objectid,
                response.status_code,
                response.text[:200] if response.text else ""
            )
            return False

        except RequestException as e:
            print(f"download error objectid={objectid}, attempt={attempt}/{max_retries}: {e}")

            if attempt < max_retries:
                time.sleep(2 * attempt)

    print(f"skipped objectid={objectid} after {max_retries} retries")
    return False

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["lat", "lon", "requested_datetime"])
    df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["requested_datetime"])

    start_date = pd.Timestamp(f"{YEAR}-01-01", tz="UTC")
    df = df[df["requested_datetime"] >= start_date]

    zip_lookup = load_zip_lookup(ZIP_GEOJSON_PATH)

    for _, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        zipcode = clean_zipcode(row.get("zipcode"))
        if zipcode is None:
            zipcode = infer_zipcode_from_point(lon, lat, zip_lookup)
        week = week_folder(row["requested_datetime"])

        image_dir = out_dir / week / zipcode
        image_dir.mkdir(parents=True, exist_ok=True)

        out_path = image_dir / make_image_name(row)
        if out_path.exists():
            continue

        params = {
            "bbox": bbox_around_point(lon, lat, half_size_m=80),
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": "512,512",
            "format": "png",
            "f": "image",
        }

        download_image_with_retry(
            URL,
            params=params,
            out_path=out_path,
            objectid=row["objectid"],
            max_retries=3
        )       

        time.sleep(0.2)


if __name__ == "__main__":
    main()
