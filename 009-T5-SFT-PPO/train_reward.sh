
# BEST IS  lr0.0005-wd0.0-ep30

wd=0.0
epochs=10
lr=0.0001
margin=25.0

python3 \
  train_reward.py \
  --db=data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.multidecode.db.backup \
  --model=data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final \
  --learning_rate=$lr \
  --weight_decay=$wd \
  --epochs=${epochs} \
  --suffix=fixed \
  --test_frac=0.05 \
  --margin=${margin} \
  --tb_suffix=lr${lr}-wd${wd}-ep${epochs}-final \
  --out_model=data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.margin${margin}-reward-model
  #--train_all_parameters=True \
