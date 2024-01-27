# Download the HTMLs you want

```
bash download-htmls.sh mythenquai 2017 01
bash download-htmls.sh tiefenbrunnen 2017 01
```

# Parse HTMLs and create numpy timeseries

```
python3 parse-and-write-timeseries.py \
  --location=mythenquai \
  --min_year=2020 \
  --output_file=mythenquai.gt2020.npy
```

# Filter days with broken data

...

# Combine data from two stations

## Generate data with `history=16`, `future=16`:

```
python3 generate-examples.py \
  --features=wind_speed,gust_speed,air_pressure,air_temp,sin_wind_dir,cos_wind_dir,sin_hour,cos_hour \
  --files=tiefenbrunnen.clean.npy,mythenquai.clean.npy \
  --output_file=both.clean.small.8feature.16h.examples \
  --history=16 \
  --future=16
```

## Generate data with `history=32`, `future=16`:

```
python3 generate-examples.py \
  --features=wind_speed,gust_speed,air_pressure,air_temp,sin_wind_dir,cos_wind_dir,sin_hour,cos_hour \
  --files=tiefenbrunnen.clean.npy,mythenquai.clean.npy \
  --output_file=both.clean.small.8feature.16h.examples \
  --history=32 \
  --future=16
```
