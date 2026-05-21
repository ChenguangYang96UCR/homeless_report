import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = "/Users/chenguangyang/Desktop/ucr_work/homeless_report"
DEFAULT_YEARS = [2022, 2023, 2024, 2025]


def read_weekly_embedding_csv(path):
    csv.field_size_limit(sys.maxsize)
    weeks = []
    arrays = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weeks.append(row["week"].replace("week_", ""))
            arrays.append(np.asarray(ast.literal_eval(row["embedding"]), dtype=np.float32))
    if not arrays:
        raise ValueError(f"No embeddings found in {path}")
    order = np.argsort(weeks)
    sorted_weeks = [weeks[i] for i in order]
    return sorted_weeks, np.stack(arrays, axis=0)[order]


def default_cases_path(root, year):
    return Path(root) / "raw_data" / str(year) / f"{year}_filtered_cases_fc_with_zip_lat_lon.csv"


def default_weekly_image_path(root, year):
    return Path(root) / "image_embedding" / f"image_embedding_cluster_{year}" / f"weekly_image_embeddings_{year}.csv"


def default_weekly_text_path(root, year):
    return Path(root) / "text_embedding" / f"text_embedding_cluster_{year}" / f"weekly_text_embeddings_{year}.csv"


def read_cases(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No case files were provided.")
    return pd.concat(frames, ignore_index=True)


def read_multi_year_embeddings(image_paths, text_paths):
    rows = []
    per_file = []
    for image_path, text_path in zip(image_paths, text_paths):
        image_weeks, image_embeddings = read_weekly_embedding_csv(image_path)
        text_weeks, text_embeddings = read_weekly_embedding_csv(text_path)
        if image_embeddings.shape[1] != text_embeddings.shape[1]:
            raise ValueError(f"Image/text slot count mismatch: {image_path} vs {text_path}")
        image_by_week = {week: image_embeddings[i] for i, week in enumerate(image_weeks)}
        text_by_week = {week: text_embeddings[i] for i, week in enumerate(text_weeks)}
        common_weeks = sorted(set(image_by_week) & set(text_by_week))
        for week in common_weeks:
            rows.append((week, image_by_week[week], text_by_week[week]))
        per_file.append(
            {
                "image_path": str(image_path),
                "text_path": str(text_path),
                "image_weeks": len(image_weeks),
                "text_weeks": len(text_weeks),
                "aligned_common_weeks": len(common_weeks),
                "image_only_weeks": sorted(set(image_weeks) - set(text_weeks)),
                "text_only_weeks": sorted(set(text_weeks) - set(image_weeks)),
                "first_week": image_weeks[0],
                "last_week": image_weeks[-1],
                "slots": int(image_embeddings.shape[1]),
            }
        )
    if not rows:
        raise ValueError("No aligned weekly embeddings found.")

    weeks = sorted({week for week, _, _ in rows})
    image_arrays = []
    text_arrays = []
    duplicate_week_counts = {}
    for week in weeks:
        matched = [(image, text) for row_week, image, text in rows if row_week == week]
        duplicate_week_counts[week] = len(matched)
        image_arrays.append(np.mean([image for image, _ in matched], axis=0))
        text_arrays.append(np.mean([text for _, text in matched], axis=0))
    duplicate_week_counts = {week: count for week, count in duplicate_week_counts.items() if count > 1}
    return weeks, np.stack(image_arrays, axis=0), np.stack(text_arrays, axis=0), per_file, duplicate_week_counts


def resize_vector(vec, target_dim):
    if target_dim is None or len(vec) == target_dim:
        return vec
    out = np.zeros(target_dim, dtype=np.float32)
    keep = min(len(vec), target_dim)
    out[:keep] = vec[:keep]
    return out


def parse_zip_order(path):
    if path is None:
        return None
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
            values.append(int(value))
    return values


def fit_normalizer(data, mode):
    data = data.astype(np.float32)
    if mode == "none":
        return {"mode": mode}
    if mode == "global":
        mean = float(data.mean())
        std = float(data.std())
        return {"mean": mean, "std": std, "mode": mode}
    if mode == "per_column":
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        return {
            "mean": mean.squeeze(0).astype(float).tolist(),
            "std": std.squeeze(0).astype(float).tolist(),
            "mode": mode,
        }
    raise ValueError(f"Unknown normalize mode: {mode}")


def apply_normalizer(data, scaler):
    data = data.astype(np.float32)
    mode = scaler.get("mode")
    if mode == "none":
        return data
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return (data - mean) / (std + 1e-8)


def split_ranges(n_rows, train_ratio, val_ratio):
    if n_rows < 3:
        raise ValueError("Need at least 3 time steps to create train/val/test splits.")
    train_end = int(round(n_rows * train_ratio))
    val_end = train_end + int(round(n_rows * val_ratio))
    train_end = min(max(train_end, 1), n_rows - 2)
    val_end = min(max(val_end, train_end + 1), n_rows - 1)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_rows),
    }


def save_npz(path, data, start_time, index):
    np.savez(
        path,
        data=data.astype(np.float32),
        start_time=start_time.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype=str),
        index=index.astype(np.int64),
    )


def write_vector_csv(path, id_column, vector_column, vectors, filename_mode=False):
    rows = []
    for i, vec in enumerate(vectors):
        key = f"{i}.png" if filename_mode else i
        rows.append({id_column: key, vector_column: json.dumps(vec.astype(float).tolist())})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Philadelphia homeless-report weekly data into the same broad file shapes as the GLA inputs."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--weekly-image", nargs="*", default=None)
    parser.add_argument("--weekly-text", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/philly_2022_2025_gla_format")
    parser.add_argument("--zip-order", default=None, help="Optional one-ZIP-per-line file matching the embedding slot order.")
    parser.add_argument("--normalize", choices=["none", "global", "per_column"], default="global")
    parser.add_argument("--text-target-dim", type=int, default=768, help="Use 0 to keep original dimension.")
    parser.add_argument("--image-target-dim", type=int, default=1000, help="Use 0 to keep original dimension.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    years_label = f"{min(args.years)}_{max(args.years)}" if len(args.years) > 1 else str(args.years[0])
    case_paths = [Path(p) for p in args.cases] if args.cases else [default_cases_path(args.root, y) for y in args.years]
    image_paths = [Path(p) for p in args.weekly_image] if args.weekly_image else [default_weekly_image_path(args.root, y) for y in args.years]
    text_paths = [Path(p) for p in args.weekly_text] if args.weekly_text else [default_weekly_text_path(args.root, y) for y in args.years]
    if not (len(case_paths) == len(image_paths) == len(text_paths)):
        raise ValueError("--cases, --weekly-image, and --weekly-text must have the same number of files.")
    for path in [*case_paths, *image_paths, *text_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    cases = read_cases(case_paths)
    cases["requested_datetime"] = pd.to_datetime(cases["requested_datetime"], errors="coerce", utc=True)
    cases = cases.dropna(subset=["requested_datetime", "zipcode"])
    cases["zip_int"] = cases["zipcode"].astype(int)
    cases["iso_week"] = cases["requested_datetime"].dt.strftime("%G-W%V")

    image_weeks, image_embeddings, text_embeddings, embedding_sources, duplicate_embedding_weeks = read_multi_year_embeddings(
        image_paths, text_paths
    )

    zip_order = parse_zip_order(args.zip_order)
    if zip_order is None:
        zip_order = sorted(cases["zip_int"].unique().tolist())

    slot_count = int(image_embeddings.shape[1])
    zip_labels = list(zip_order)
    if len(zip_labels) < slot_count:
        zip_labels.extend([f"UNMAPPED_SLOT_{i}" for i in range(len(zip_labels), slot_count)])
    elif len(zip_labels) > slot_count:
        raise ValueError(
            f"ZIP order has {len(zip_labels)} entries but embeddings have only {slot_count} slots."
        )

    weeks = image_weeks
    counts = (
        cases.groupby(["iso_week", "zip_int"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=weeks, columns=zip_labels, fill_value=0)
    )
    case_zip_set = set(cases["zip_int"].unique().tolist())
    numeric_zip_labels = {z for z in zip_labels if isinstance(z, int)}
    case_zips_not_in_embedding = sorted(case_zip_set - numeric_zip_labels)
    embedding_zips_without_cases = sorted(numeric_zip_labels - case_zip_set)
    raw_data = counts.to_numpy(dtype=np.float32)
    start_time = pd.to_datetime([f"{w}-1" for w in weeks], format="%G-W%V-%u", utc=True)
    index = np.arange(raw_data.shape[1], dtype=np.int64)
    ranges = split_ranges(len(raw_data), args.train_ratio, args.val_ratio)
    scaler = fit_normalizer(raw_data[slice(*ranges["train"])], args.normalize)
    data = apply_normalizer(raw_data, scaler)
    split_outputs = {}
    for split_name, (start, end) in ranges.items():
        split_path = output_dir / f"philly_{years_label}_{split_name}1.npz"
        save_npz(split_path, data[start:end], start_time[start:end], index)
        split_outputs[split_name] = str(split_path)
    full_path = output_dir / f"philly_{years_label}_full.npz"
    save_npz(full_path, data, start_time, index)

    text_target_dim = None if args.text_target_dim == 0 else args.text_target_dim
    image_target_dim = None if args.image_target_dim == 0 else args.image_target_dim
    text_static = text_embeddings.mean(axis=0)
    image_static = image_embeddings.mean(axis=0)
    text_static = np.stack([resize_vector(v, text_target_dim) for v in text_static])
    image_static = np.stack([resize_vector(v, image_target_dim) for v in image_static])

    text_csv_path = output_dir / f"philly_{years_label}_poi_vectors.csv"
    image_csv_path = output_dir / f"philly_{years_label}_image_features.csv"
    write_vector_csv(text_csv_path, "list_id", "sentence_vector", text_static)
    write_vector_csv(image_csv_path, "filename", "feature_vector", image_static, filename_mode=True)

    manifest = {
        "outputs": {
            "npz": split_outputs,
            "full_npz": str(full_path),
            "text_csv": str(text_csv_path),
            "image_csv": str(image_csv_path),
        },
        "years": args.years,
        "case_files": [str(p) for p in case_paths],
        "embedding_sources": embedding_sources,
        "duplicate_embedding_weeks_averaged": duplicate_embedding_weeks,
        "counts_shape": list(data.shape),
        "split_shapes": {
            name: list(data[start:end].shape) for name, (start, end) in ranges.items()
        },
        "split_ranges": {
            name: {
                "row_start": start,
                "row_end_exclusive": end,
                "start_time": str(start_time[start]),
                "end_time": str(start_time[end - 1]),
            }
            for name, (start, end) in ranges.items()
        },
        "embedding_week_slot_shape": list(image_embeddings.shape[:2]),
        "zip_order": zip_labels,
        "normalization": scaler,
        "notes": [
            "NPZ data is weekly request count by ZIP, reindexed to the weekly embedding order.",
            "Text/image CSVs are static per-slot means over all included weeks, matching the static shape of the GLA CSV inputs.",
            "If embedding slots are not in sorted ZIP order, pass --zip-order with the real slot-to-ZIP mapping.",
            "Cases whose ZIP is absent from the embedding ZIP order are excluded by the reindexing step.",
            "Embedding ZIPs with no included cases are retained as all-zero count columns.",
            "Text vectors are resized to 768 and image vectors to 1000 by default to mimic the GLA file dimensions.",
        ],
        "zip_alignment": {
            "case_zips_not_in_embedding": case_zips_not_in_embedding,
            "embedding_zips_without_cases": embedding_zips_without_cases,
        },
        "raw_dimensions": {
            "weekly_text_embedding_dim": int(text_embeddings.shape[2]),
            "weekly_image_embedding_dim": int(image_embeddings.shape[2]),
            "output_text_embedding_dim": int(text_static.shape[1]),
            "output_image_embedding_dim": int(image_static.shape[1]),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
