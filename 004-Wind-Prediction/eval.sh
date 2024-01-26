for NUM_CONVS in 0 1 2; do
  for CHANNELS in 10 20; do
    for CONV_LEN in 4 8; do
      for DENSE_SIZE in 20 40 80; do
        for NUM_DENSE in 1 2 3; do
          python3 train.py \
            --debug_every_percent=5 \
            --num_convs=${NUM_CONVS} \
            --conv_len=8 \
            --down_scale=2 \
            --channels=${CHANNELS} \
            --num_dense=${NUM_DENSE} \
            --dense_size=${DENSE_SIZE} \
            --learning_rate=0.001 \
            --epochs=20.0 \
            --batch_size=128 \
            --data=data/both.clean.small.8feature.16h.examples.npz \
            --model_name=models/mini \
            --model=cnn \
            --batch_norm=True \
            --padding=SAME \
            --prefix=real
          done
        done
      done
    done
  done
