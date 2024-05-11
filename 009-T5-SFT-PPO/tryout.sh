BASE="python3 train_reward.py --db=data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.multidecode.db.backup --model=data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final"

# BEST IS  lr0.0005-wd0.0-ep30

for wd in 0.0; do
  for epochs in 5 10; do
    for lr in 0.0001 0.0005 0.001 0.005; do
      ${BASE} \
        --learning_rate=$lr \
        --weight_decay=$wd \
        --epochs=${epochs} \
        --suffix=fixed \
        --train_all_parameters=True \
        --test_frac=0.05 \
        --tb_suffix=lr${lr}-wd${wd}-ep${epochs}-final
    done
  done
done


