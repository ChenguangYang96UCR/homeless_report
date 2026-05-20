import csv
from pathlib import Path

YEAR=2025
src = Path(f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/raw_data/{YEAR}/{YEAR}_filtered_cases_fc.csv")
out = Path(f"/Users/chenguangyang/Desktop/ucr_work/homeless_report/raw_data/{YEAR}/{YEAR}_filtered_cases_fc_with_zip_lat_lon.csv")

kept = 0
total = 0

with src.open("r", newline="", encoding="utf-8-sig") as f, out.open("w", newline="", encoding="utf-8") as g:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(g, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        total += 1

        zipcode = (row.get("zipcode") or "").strip()
        lat = (row.get("lat") or "").strip()
        lon = (row.get("lon") or "").strip()

        if not zipcode or not lat or not lon:
            continue

        try:
            float(lat)
            float(lon)
        except ValueError:
            continue

        writer.writerow(row)
        kept += 1

print(f"original line number: {total}")
print(f"keep line number: {kept}")
print(f"output file name: {out}")