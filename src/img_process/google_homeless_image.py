import os
import time
import pandas as pd
import requests

CSV_PATH = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/raw_data/filter/2020/2020_filtered_cases_fc.csv"
OUT_DIR = "/Users/chenguangyang/Desktop/ucr_work/homeless_report/satellite_images/2020"

GOOGLE_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 只保留有经纬度的记录
df = df.dropna(subset=["lat", "lon"])

for _, row in df.iterrows():
    objectid = row["objectid"]
    lat = float(row["lat"])
    lon = float(row["lon"])

    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": 20,
        "size": "640x640",
        "maptype": "satellite",
        "key": GOOGLE_API_KEY,
    }

    out_path = os.path.join(OUT_DIR, f"{objectid}.png")

    if os.path.exists(out_path):
        continue

    response = requests.get(url, params=params, timeout=30)

    if response.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f"saved: {out_path}")
    else:
        print(f"failed objectid={objectid}, status={response.status_code}, text={response.text[:200]}")

    # 避免请求太快
    time.sleep(0.1)