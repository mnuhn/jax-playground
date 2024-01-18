from bs4 import BeautifulSoup
import glob
import io
import numpy as np
from datetime import datetime
from tqdm import tqdm

date_format = "%d.%m.%Y %H:%M:%S"

field_names = ["date", "airtemp", "humidity", "gust_speed", "wind_speed", "wind_strength", "wind_dir", "wind_chill", "water_temp", "air_pressure", "dew_point"]

fs = sorted(glob.glob("./tiefenbrunnen/20??-??.html"))

data = []

def extract_ym(fn):
  fn = fn.split("/")[-1]
  year = int(fn.split("-")[0])
  month = int(fn.split("-")[1].split(".")[0])

  assert month > 0
  assert month < 13
  assert year > 2019
  assert year < 2025
  return year, month

for f in tqdm(fs):
  year, month = extract_ym(f)
  f = io.open(f, encoding="utf-8", errors='replace')
  soup = BeautifulSoup(f,features="lxml" )
  tables = soup.find_all('table')
  table = tables[1]
  rows = table.find_all('tr')
  first = True
  for row in rows:
    # Extract data from each cell in the row
    cells = row.find_all(['th', 'td'])
    cells = [cell.text.strip() for cell in cells]
    assert len(cells) == len(field_names)
    if not first:
      d = datetime.strptime(cells[0], date_format)
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
data = np.array(data)
np.save("./timeseries", data)
