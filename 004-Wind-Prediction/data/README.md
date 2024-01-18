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
