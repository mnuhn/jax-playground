
# BEST IS  lr0.0005-wd0.0-ep30
# margin25-lr0.0001-wd0.0-ep20-trainallparamsTrue-final

# BEST IS margin25-lr0.001-wd0.0-ep5-trainallparamsFalse-final
# See ./reward-model-comparison.tensorboard

wd=0.0
epochs=2
lr=0.0001
margin=25.0
# margin25-lr0.0005-wd0.0-ep5-trainallparamsTrue-final
#
python3 \
  train_reward.py \
  --db=data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.multidecode.db.backup \
  --model=data/mix22.extended.txt.model2-final \
  --learning_rate=$lr \
  --weight_decay=$wd \
  --epochs=${epochs} \
  --suffix=fixed \
  --test_frac=0.05 \
  --margin=${margin} \
  --tb_suffix=lr${lr}-wd${wd}-ep${epochs}-final \
  --out_model=data/mix22.extended.txt.model2-final.lr${lr}-wd${wd}-ep${epochs}-margin${margin}-reward-model-head4

#python3 \
#  train_reward.py \
#  --db=data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.multidecode.db.backup \
#  --model=data/mix22.extended.txt.model2-final \
#  --learning_rate=$lr \
#  --weight_decay=$wd \
#  --epochs=${epochs} \
#  --train_all_parameters=True \
#  --suffix=fixed \
#  --test_frac=0.05 \
#  --margin=${margin} \
#  --tb_suffix=lr${lr}-wd${wd}-ep${epochs}-final \
#  --out_model=data/mix22.extended.txt.model2-final2.lr${lr}-wd${wd}-ep${epochs}-margin${margin}-reward-model-allparams
