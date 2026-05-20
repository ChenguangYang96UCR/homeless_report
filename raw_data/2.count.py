import pandas as pd

time = 2020
df = pd.read_csv(f'{time}/{time}_filtered_cases_fc.csv')

# -----------------------------
# Basic cleaning
# -----------------------------
df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")

df["date"] = df["requested_datetime"].dt.date
df["week"] = df["requested_datetime"].dt.to_period("W").astype(str)
df["year"] = df["requested_datetime"].dt.year

df["zipcode"] = df["zipcode"].astype("string").str.strip()
df["address"] = df["address"].astype("string").str.strip()
df["service_name"] = df["service_name"].astype("string").str.strip()

# -----------------------------
# 1. Count homeless happen number by zipcode
# No service_name / subject filtering
# -----------------------------
df_with_zip = df[
    df["zipcode"].notna() &
    (df["zipcode"] != "")
].copy()

daily_zip_counts = (
    df_with_zip
    .groupby(["date", "zipcode"])
    .size()
    .reset_index(name="homeless_count")
    .sort_values(["date", "zipcode"])
)

weekly_zip_counts = (
    df_with_zip
    .groupby(["week", "zipcode"])
    .size()
    .reset_index(name="homeless_count")
    .sort_values(["week", "zipcode"])
)

# -----------------------------
# 1.1 Yearly count for each zipcode
# -----------------------------
yearly_zip_counts = (
    df_with_zip
    .groupby(["year", "zipcode"])
    .size()
    .reset_index(name="homeless_count")
    .sort_values(["year", "zipcode"])
)

daily_zip_counts.to_csv(f'{time}/{time}_homeless_daily_by_zipcode.csv', index=False)
weekly_zip_counts.to_csv(f'{time}/{time}_homeless_weekly_by_zipcode.csv', index=False)
yearly_zip_counts.to_csv(f'{time}/{time}_homeless_yearly_by_zipcode.csv', index=False)


# -----------------------------
# 2. Check whether ONLY Information Request has no address
# -----------------------------
df["has_address"] = (
    df["address"].notna() &
    (df["address"] != "")
)

no_address_df = df[~df["has_address"]].copy()

# Summary by service_name among records without address
no_address_service_summary = (
    no_address_df
    .groupby("service_name")
    .size()
    .reset_index(name="no_address_count")
    .sort_values("no_address_count", ascending=False)
)

# Check result: whether all no-address records are Information Request
only_info_request_no_address = (
    len(no_address_df) > 0 and
    no_address_df["service_name"].eq("Information Request").all()
)

check_summary = pd.DataFrame({
    "total_records": [len(df)],
    "records_without_address": [len(no_address_df)],
    "only_information_request_has_no_address": [only_info_request_no_address],
    "non_information_request_without_address": [
        (~no_address_df["service_name"].eq("Information Request")).sum()
    ]
})

check_summary.to_csv(
    f'{time}/{time}_check_only_information_request_no_address.csv',
    index=False
)

print("Saved:")
print(f"1. {time}/{time}_homeless_daily_by_zipcode.csv")
print(f"2. {time}/{time}_homeless_weekly_by_zipcode.csv")
print(f"3. {time}/{time}_homeless_yearly_by_zipcode.csv")
print(f"4. {time}/{time}_check_only_information_request_no_address.csv")

print("\nYearly zipcode counts:")
print(yearly_zip_counts)

print("\nCheck summary:")
print(check_summary)

print("\nNo-address records by service_name:")
print(no_address_service_summary)