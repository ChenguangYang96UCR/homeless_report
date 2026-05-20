import pandas as pd

############################################################
# filter cases based on keyword
############################################################
time = 2024

data = pd.read_csv(f'{time}/{time}_public_cases_fc.csv')
subject_col = data['subject']

keyword = "homeless"
filtered_data = data[data["subject"].str.contains(keyword, case=False, na=False)]

filtered_data.to_csv(f'{time}/{time}_filtered_cases_fc.csv', index=False)

############################################################
# Sperate the 
############################################################
data = pd.read_csv(f"{time}/{time}_filtered_cases_fc.csv")

data['requested_datetime'] = pd.to_datetime(data['requested_datetime'])

data = data.sort_values(by='requested_datetime').reset_index(drop=True)

print(f"{data['requested_datetime']}")
print(data["requested_datetime"].is_monotonic_increasing)


# weekly summary record
data["week_start"] = data["requested_datetime"].dt.normalize() - pd.to_timedelta(data["requested_datetime"].dt.weekday, unit="D")

data["week_end"] = data["week_start"] + pd.Timedelta(days=6)

print(data[["requested_datetime", "week_start", "week_end"]].head(20))

weekly_counts = data.groupby("week_start").size().reset_index(name="count")

weekly_counts.to_csv(f"{time}/{time}_weekly_counts.csv", index=False)

print(weekly_counts)

# dialy summary record
data["date"] = data["requested_datetime"].dt.floor("D")

daily_counts = data.groupby("date").size().rename("count")


full_dates = pd.date_range(start=data["date"].min(), end=data["date"].max(), freq="D")
daily_counts = daily_counts.reindex(full_dates, fill_value=0).reset_index()
daily_counts.columns = ["date", "count"]
daily_counts.to_csv(f"{time}/{time}_daily_counts.csv", index=False)