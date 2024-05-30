python3 train_ppo.py \
  --prompts_fn=data/ppo_mix.shuffled.txt \
  --model=./data/mix23.txt.augmented2.model-final \
  --max_ppo_steps=150 \
  --max_len=100 \
  --batch_size=128 \
  --target_kl=6.0 \
  --learning_rate=0.000015 \
  --rule_reward_fac=0.01 \
  --suffix=reward-mixing13-augmented2 \
  --low_rule_reward_override=True \
  --use_score_scaling=True \
  --reward_model=./data/mix22.extended.txt.model2-final.lr0.0001-wd0.0-ep2-margin25.0-reward-model-head4-final
  # ./data/mix22.extended.txt.model2-final.lr0.001-wd0.0-ep5-margin25.0-reward-model-final


exit 0


# mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.margin25.0-reward-model-final/ \
# ./data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.lr0.0001-wd0.0-ep3-margin25.0-reward-model-final
for horizon in 10000; do
  for batch_size in 128 256; do
    for rule_reward_fac in 0.3 0.5 0.9 0.8 0.7 0.6; do #0.05 0.1 0.3 0.5; do
      for score_scaling_flag in "" "--use_score_scaling=True"; do
        for score_norm in "" "--use_score_norm=True"; do
            #--init_kl_coef=5.0 \
            #--kl_horizon=${horizon} \
          python3 train_ppo.py \
            --prompts_fn=data/ppo_mix.shuffled.txt \
            --model=./data/mix23.txt.model-final \
            --max_ppo_steps=200 \
            --max_len=100 \
            --batch_size=${batch_size} \
            --target_kl=6.0 \
            --learning_rate=0.000015 \
            ${score_scaling_flag} \
            --rule_reward_fac=${rule_reward_fac} \
            --suffix=reward-mixing \
            --reward_model=./data/mix22.extended.txt.model2-final.lr0.001-wd0.0-ep5-margin25.0-reward-model-final
        done
      done
    done
  done
done
