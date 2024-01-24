import numpy as np
import argparse
import datetime
import os

def file_doesnt_exist(fn):
  if os.path.exists(fn):
    raise argparse.ArgumentTypeError(f"File '{fn}' already exists.")
  return fn

parser = argparse.ArgumentParser(description="Process the location option.")
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--output_file", type=file_doesnt_exist, required=True)
parser.add_argument("--min_year", type=int, default=None)
parser.add_argument("--max_year", type=int, default=None)
args = parser.parse_args()

FIELDS = ["year", "month", "day", "hour", "minute", "seconds", "air_temp",
          "humidity", "gust_speed", "wind_speed", "wind_strength", "wind_dir",
          "wind_chill", "water_temp", "air_pressure", "dew_point"]

data = np.load(args.file)

def skip(year):
  if args.min_year and year < args.min_year:
    return True
  if args.max_year and year > args.max_year:
    return True
  return False

def check_column(data, col_name, min_non_nan_percent = 99, min_non_zero_percent = 90, min_min = None, max_max = None):
  col = data[:,FIELDS.index(col_name)]
  non_nan = np.count_nonzero(~np.isnan(col))
  non_nan_percent = 100.0 * non_nan / len(col)
  non_zero = np.count_nonzero(col)
  non_zero_percent = 100.0 * non_zero / len(col)
  col_min = np.min(col)
  col_max = np.max(col)

  ok = True
  ok &= not min_min or col_min >= min_min
  ok &= not max_max or col_max <= max_max
  ok &= not min_non_nan_percent or non_nan_percent > min_non_nan_percent
  ok &= not min_non_zero_percent or non_zero_percent > min_non_zero_percent

  if not ok:
    print(f"  {col_name}: non-nan: {non_nan_percent:.1f}%, non-zero: {non_zero_percent:.1f}%, min: {col_min}, max: {col_max}")

  return ok

print("Total entries:", len(data))

def condition(data, year, month, day):
  condition = (data[:, 0] == year)
  if month:
    condition &= (data[:, 1] == month)
  if day:
    condition &= (data[:, 2] == day)

  return condition


def inspect(filtered_data):
  if len(filtered_data) == 0:
    return False
  ok = True
  ok &= check_column(filtered_data, "day", min_min = 1, max_max = 31.0)
  ok &= check_column(filtered_data, "hour", min_min = 0, max_max = 24.0)
  ok &= check_column(filtered_data, "air_temp", min_min = -15.0, max_max = 50.0)
  ok &= check_column(filtered_data, "water_temp", min_min = 0.0, max_max = 50.0)
  ok &= check_column(filtered_data, "humidity", min_min = 0.0, max_max = 105.0)
  ok &= check_column(filtered_data, "wind_speed", min_non_zero_percent = 20.0, min_min = 0.0, max_max = 35.0)
  ok &= check_column(filtered_data, "gust_speed", min_non_zero_percent = 20.0, min_min = 0.0, max_max = 35.0)
  ok &= check_column(filtered_data, "wind_dir", min_non_zero_percent = 20.0, min_min = 0, max_max = 360)
  ok &= check_column(filtered_data, "air_pressure", min_min = 800, max_max = 1200)

  return ok

def build_dates(data):
  d = np.ulonglong(data)
  result = d[:, 5]
  result += d[:, 4] * 60
  result += d[:, 3] * 3600
  result += d[:, 2] * 3600 * 24
  result += d[:, 1] * 3600 * 24 * 32
  result += d[:, 0] * 3600 * 24 * 32 * 365
  return result


# TODO: minimum index
dates = build_dates(data)
deltas = dates[1:] - dates[:-1]

if not np.all(deltas >= 600):
  print("non monotonous data")


min_idx = np.argmin(dates)
max_idx = np.argmax(dates)

start = datetime.date(int(data[min_idx,0]), int(data[min_idx,1]), int(data[min_idx,2]))
end = datetime.date(int(data[max_idx,0]), int(data[max_idx,1]), int(data[max_idx,2]))

d = start

output = None
print(f"data from {start} to {end}")
keep = np.full((data.shape[0]), False)

while d <= end:
  d += datetime.timedelta(days=1)
  if skip(d.year):
    print(f"skipping {d} due to selected year range [{args.min_year}, {args.max_year}]")
    continue
  c = condition(data, d.year, d.month, d.day)
  ok = inspect(data[c])
  if ok:
    keep |= c
  elif len(data[c]) == 0:
    print(f"no data for {d}")
  else:
    print(f"skipping {d} due to data quality")

with open(args.output_file, "wb") as out:
  np.save(out, data[keep])
