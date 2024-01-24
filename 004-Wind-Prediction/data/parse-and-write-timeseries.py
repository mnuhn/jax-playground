from bs4 import BeautifulSoup
import glob
import io
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse
import os

def file_doesnt_exist(fn):
  if os.path.exists(fn):
    raise argparse.ArgumentTypeError(f"File '{fn}' already exists.")
  return fn

parser = argparse.ArgumentParser(description="Process the location option.")
parser.add_argument("--location", type=str, choices=["tiefenbrunnen", "mythenquai"], required=True)
parser.add_argument("--output_file", type=file_doesnt_exist, required=True)
args = parser.parse_args()

DATE_FORMAT = "%d.%m.%Y %H:%M:%S"

FIELD_NAMES = ["date", "airtemp", "humidity", "gust_speed", "wind_speed",
               "wind_strength", "wind_dir", "wind_chill", "water_temp",
               "air_pressure", "dew_point"]

def extract_ym(fn):
  fn = fn.split("/")[-1]
  year = int(fn.split("-")[0])
  month = int(fn.split("-")[1].split(".")[0])
  assert month > 0 and month < 13 and year > 2000 and year < 2030

  return year, month

def parse_html(fn):
  data = []

  f = io.open(fn, encoding="utf-8", errors='replace')
  soup = BeautifulSoup(f,features="lxml" )
  tables = soup.find_all('table')
  table = tables[1]
  rows = table.find_all('tr')
  first = True
  for row in rows:
    # Extract data from each cell in the row
    cells = row.find_all(['th', 'td'])
    cells = [cell.text.strip() for cell in cells]
    assert len(cells) == len(FIELD_NAMES)
    if not first:
      d = datetime.strptime(cells[0], DATE_FORMAT)
      cells = [d.year, d.month, d.day, d.hour, d.minute, d.second] + cells[1:]
      for i in range(1,len(cells)):
        cells[i] = float(cells[i])

      if cells[0] != year:
        continue
      if cells[1] != month:
        continue
      #cells = np.array(cells, dtype=np.float32)
      data.append(cells)
    first = False
  return data

fns = sorted(glob.glob(f"./{args.location}/20??-??.html"))

print(f"Found {len(fns)} files.")

data = []

for fn in tqdm(fns):
  year, month = extract_ym(fn)

  cur_data = parse_html(fn)
  print(f"found {len(cur_data)} entries in {fn}")

  data.extend(cur_data)

data = np.array(data)
np.save(args.output_file, data)
