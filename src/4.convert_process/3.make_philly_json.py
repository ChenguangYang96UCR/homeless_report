import json
from pathlib import Path

import pandas as pd


GEOJSON = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/src/pa_zipcodes.geojson"
ZIP_ORDER = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/text_embedding/text_embedding_cluster_2025/philadelphia_zipcode_order.csv"
OUTPUT = "outputs/philly_2022_2025_gla_format/philly.json"


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


def main():
    zipcodes = parse_zip_order(ZIP_ORDER)
    with open(GEOJSON) as f:
        geojson = json.load(f)
    feature_by_zip = {feature["properties"]["ZCTA5CE10"]: feature for feature in geojson["features"]}

    out = {}
    missing = []
    for idx, zipcode in enumerate(zipcodes):
        feature = feature_by_zip.get(zipcode)
        if feature is None:
            missing.append(zipcode)
            continue
        props = feature["properties"]
        out[str(idx)] = {
            "lat": float(props["INTPTLAT10"]),
            "lon": float(props["INTPTLON10"]),
        }

    output = Path(OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"wrote {output} nodes={len(out)} missing={missing}")


if __name__ == "__main__":
    main()
