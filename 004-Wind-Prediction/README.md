# Data Preparation
## Download the HTMLs you want

```
bash download-htmls.sh mythenquai 2017 01
bash download-htmls.sh tiefenbrunnen 2017 01
```

## Parse HTMLs and create numpy timeseries

```
python3 parse-and-write-timeseries.py \
  --location=mythenquai \
  --output_file=mythenquai.raw.npy
```

## Filter days with broken data

`check-data.py` runs over data for each day and throws out days for which there
is at least 1 entry NOT fullfilling the folowing conditions:

Field        | What       |   Min | Max
-------------|------------|------:|------:
DAY          | Value      |   1.0 |   31.0
HOUR         | Value      |   0.0 |   24.0
AIR_TEMP     | Value      | -15.0 |   50.0
WATER_TEMP   | Value      |   0.0 |   50.0
HUMIDITY     | Value      |   0.0 |  105.0
WIND_SPEED   | % non-zero |  20.0 |  100.0
WIND_SPEED   | Value      |   0.0 |   35.0
GUST_SPEED   | Value      |   0.0 |   35.0
WIND_DIR     | % non-zero |  20.0 |  100.0
WIND_DIR     | Value      |   0.0 |  360.0
AIR_PRESSURE | Value      | 800.0 | 1200.0

```
python3 check-data.py \
  --file=mythenquai.raw.npy \
  --output_file=./mythenquai.clean.npy
```

## Preprocessing

Data is preprocessed using `preprocess.py`. In particular, all features are
scaled to have a range of [0,1]. For this, the following input ranges are
scaled to [0,1]:

Feature      |  Unit |  From | To
-------------|------:|------:|-----:
WIND_SPEED   | m/s   |  0.0   | 25.0
GUST_SPEED   | m/s   | 0.0   | 25.0
AIR_PRESSURE | hPa   | 950.0 | 1050.0
AIR_TEMP     | deg C | -15.0 | 40.0
WATER_TEMP   | def C |0.0   | 30.0
SIN_HOUR     | n/a   | -1.0  | 1.0
COS_HOUR     | n/a   | -1.0  | 1.0
SIN_MONTH    | n/a   | -1.0  | 1.0
COS_MONTH    | n/a   | -1.0  | 1.0
SIN_WIND_DIR | n/a   | -1.0  | 1.0
COS_WIND_DIR | n/a   | -1.0  | 1.0

## Generate final train and test data

The following tool generates the final train and test data. It takes the
following parameters:

Parameter   | Description
------------|-----------------------------------------------------------
Features    | A comma-separated list of features to include as input `X`
History     | Length of history for each example in `X`
Predictions | Number of predictions to include in `Y`

The generated data are numpy `npz` files having the following arrays:

Array | Shape                | Description
------|----------------------|-----------------------------------
X     | [N,History,Features] | N is the number of train examples
Y     | [N,Predictions]      |
XT    | [M,History,Features] | M is the number of test examples
YT    | [M,Predictions]      |

Only examples that have no gap in the input training data are used as examples.

Note that the generate data is quite redundant, as can be seen in this example
(with Features=1, History=4, Future=2). Unnecessary nesting is removed:

X | Y
--|---
[1, 2, 3, 4] | [5, 6]
[2, 3, 4, 5] | [6, 7]
[3, 4, 5, 6] | [7, 8]
...|...

### Generate data with `history=16`, `future=16`:

We call this one `both.clean.small.8feature.16h.examples.npz`:

```
python3 generate-examples.py \
  --features=wind_speed,gust_speed,air_pressure,air_temp,sin_wind_dir,cos_wind_dir,sin_hour,cos_hour \
  --files=tiefenbrunnen.clean.npy,mythenquai.clean.npy \
  --output_file=both.clean.small.8feature.16h.examples \
  --history=16 \
  --future=16
```

### Generate data with `history=32`, `future=16`:

We call this one `both.clean.small.8feature.32h.examples.npz`:

```
python3 generate-examples.py \
  --features=wind_speed,gust_speed,air_pressure,air_temp,sin_wind_dir,cos_wind_dir,sin_hour,cos_hour \
  --files=tiefenbrunnen.clean.npy,mythenquai.clean.npy \
  --output_file=both.clean.small.8feature.32h.examples \
  --history=32 \
  --future=16
```

# Baselines

For `both.clean.small.8feature.16h.examples.npz`:

History | Algorithm        | sqrt(MSE)
--------|------------------|-----------
n/a     |       last_value | 0.04641434
n/a     | const_value_0.00 | 0.09431773
n/a     | const_value_0.07 | 0.06256304
16      |       mean_value | 0.04707646
32      |       mean_value | 0.04519560
