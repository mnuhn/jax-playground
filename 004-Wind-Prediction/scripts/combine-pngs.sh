
mkdir -p models/combined
for NUM in `seq --format '%04g' 0 2 1000`; do
  echo ${NUM}
  convert \
  models/train.50000.sin_hour.cos_hour.flax.png/${NUM}.png \
  models/train.50000.wind_speed.flax.png/${NUM}.png \
  models/train.50000.gust_speed.flax.png/${NUM}.png \
  -append models/combined/${NUM}.png
#  models/train.50000.air_pressure.flax.png/${NUM}.png \
#  models/train.50000.air_temp.flax.png/${NUM}.png \
done
