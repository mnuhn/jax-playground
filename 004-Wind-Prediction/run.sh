ITERS=50000
EVERY_PERCENT=5
C=20
BS=64

FEATURES=""

for FEATURES in sin_hour,cos_hour,wind_speed,gust_speed,sin_wind_dir,cos_wind_dir,air_pressure,air_temp,water_temp; do
  #FEATURES="${FEATURES},${FEATURE}"
  #FEATURES="${FEATURES#,}"
  MODEL="models/train.${ITERS}.$(echo "${FEATURES}" | tr ',' '.').flax"

  echo ${FEATURES}
  echo ${MODEL}

  python3 train.py \
    --debug_every_percent=50 \
    --history=257 \
    --conv_len=16 \
    --down_scale=2 \
    --channels=20 \
    --dense_size=250 \
    --learning_rate=0.001 \
    --iters=50000 \
    --batch_size=64 \
    --pred=120 \
    --timeseries=timeseries.lt2023.npy \
    --features="${FEATURES}" \
    --model_name="${MODEL}"

  echo python3 predict.py \
    --debug_every_percent=50 \
    --history=257 \
    --conv_len=16 \
    --down_scale=2 \
    --channels=20 \
    --dense_size=250 \
    --learning_rate=0.001 \
    --iters=50000 \
    --batch_size=2 \
    --pred=120 \
    --timeseries=timeseries.2023.npy \
    --num=1000 \
    --features="${FEATURES}" \
    --model_name="${MODEL}" &
done

