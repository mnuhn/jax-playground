import numpy as np
import jax.numpy as jnp

FIELDS = ["year", "month", "day", "hour", "minute", "seconds", "air_temp",
          "humidity", "gust_speed", "wind_speed", "wind_strength", "wind_dir",
          "wind_chill", "water_temp", "air_pressure", "dew_point"]

YEAR = 0
MONTH = 1
DAY = 2
HOUR = 3
MINUTE = 4
SECONDS = 5
AIR_TEMP = 6
HUMIDITY = 7
GUST_SPEED = 8
WIND_SPEED = 9
WIND_STRENGTH = 10
WIND_DIR = 11
WIND_CHILL = 12
WATER_TEMP = 13
AIR_PRESSURE = 14
DEW_POINT = 15

# Extended fields
FIELDS.extend(["datetime", "sin_hour", "cos_hour", "sin_wind_dir", "cos_wind_dir"])

SIN_HOUR = 16
COS_HOUR = 17
SIN_WIND_DIR = 18
COS_WIND_DIR = 19

DATETIME = 20
GAP = 21

def range_to_unit(data, y_min, y_max):
  assert y_min < y_max
  scaled = (data - y_min) / (y_max - y_min)
  scaled = np.clip(scaled, 0.0, 1.0)
  return scaled

def get_date(row):
  return datetime.datetime(year=int(row[0]), month=int(row[1]),
                    day=int(row[2]), hour=int(row[3]),
                    minute=int(row[4]), second=int(row[5]))


def preprocess_features(data):
  # Preprocess
  data[:,WIND_SPEED] = range_to_unit(data[:,WIND_SPEED], 0.0, 25.0)
  data[:,GUST_SPEED] = range_to_unit(data[:,GUST_SPEED], 0.0, 25.0)
  data[:,AIR_PRESSURE] = range_to_unit(data[:,AIR_PRESSURE], 950.0, 1050.0)
  data[:,AIR_TEMP] = range_to_unit(data[:,AIR_TEMP], -15.0, 40.0)
  data[:,WATER_TEMP] = range_to_unit(data[:,WATER_TEMP], 0.0, 30.0)

  # Add 2 new fields.
  data = np.c_[data, data[:,HOUR]]
  data = np.c_[data, data[:,HOUR]]
  data[:,SIN_HOUR] = jnp.sin(2.0 * np.pi * data[:, SIN_HOUR] / 24.0)
  data[:,COS_HOUR] = jnp.cos(2.0 * np.pi * data[:, COS_HOUR] / 24.0)

  # Add 2 new fields.
  data = np.c_[data, data[:,WIND_DIR]]
  data = np.c_[data, data[:,WIND_DIR]]
  data[:,SIN_WIND_DIR] = jnp.sin(2.0 * np.pi * data[:, SIN_WIND_DIR] / 360.0)
  data[:,COS_WIND_DIR] = jnp.cos(2.0 * np.pi * data[:, COS_WIND_DIR] / 360.0)
  data[:,WIND_DIR] = range_to_unit(data[:,WIND_DIR], 0.0, 359.0)

  # Add 1 new field.
  data = np.c_[data, data[:,YEAR]]
  data = np.c_[data, data[:,YEAR]]

  # Compute all positions where data is not 10m apart.
  dates = np.array([f"{int(year)}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
                      for year, month, day, hour, minute, second in data[:, :6]], dtype='datetime64[s]')
  differences = np.diff(dates)
  prepend = np.array([np.timedelta64(0, 'm')], dtype='timedelta64[m]')
  differences = np.concatenate([prepend, differences])

  data[:,DATETIME] = dates
  data[:, GAP] = (differences != np.timedelta64(10, 'm'))

  data[:,HOUR] = range_to_unit(data[:,HOUR], 0.0, 24.0)
  data[:,MINUTE] = range_to_unit(data[:,MINUTE], 0.0, 59.0)
  data[:,SECONDS] = range_to_unit(data[:,SECONDS], 0.0, 59.0)

  return data



