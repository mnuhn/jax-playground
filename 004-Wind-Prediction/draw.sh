python3 \
  draw.py \
  --history=257 \
  --predictions=120 \
  --timeseries=timeseries.2023.npy \
  --prediction_file=predictions/sin_hour.cos_hour.wind_speed.gust_speed.sin_wind_dir.cos_wind_dir.air_pressure.air_temp.water_temp.npy,predictions/sin_hour.cos_hour.npy,predictions/water_temp.npy,predictions/air_temp.npy,predictions/air_pressure.npy,predictions/wind_speed.npy,predictions/gust_speed.npy,predictions/sin_wind_dir.cos_wind_dir.npy \
  --prediction_labels="all combined,hour,water_temp,air temp,air pressure,wind speed,gust speed,wind direction"
