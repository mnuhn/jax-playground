for wd in 0.0; do
  for epochs in 5 10 20; do
    for lr in 0.00005 0.0001 0.0005 0.001 0.005; do
      for margin in 25; do # 10 5 1 0; do
        for train_all_parameters in True False; do
          if [ ${train_all_parameters} == "True" ]; then
            TRAIN_ALL_PARAMETERS_EXTRA="--train_all_parameters=True"
          else
            TRAIN_ALL_PARAMETERS_EXTRA=""
          fi
          python3 train_reward.py \
            --db=./data/mix24.reward.merged_ratings.db \
            --model=./data/mix23.txt.model-final \
            --out_model=data/mix24.reward.merged_ratings.db.scan-reward \
            --margin=${margin} \
            --learning_rate=$lr \
            --weight_decay=$wd \
            --epochs=${epochs} \
            --suffix=fixed \
            --test_frac=0.10 \
            --tb_suffix=margin${margin}-lr${lr}-wd${wd}-ep${epochs}-trainallparams${train_all_parameters}-final \
            ${TRAIN_ALL_PARAMETERS_EXTRA}
          done
      done
    done
  done
done


