for horizon in 10000; do
  for rule_reward_fac in 0.1; do #0.05 0.1 0.3 0.5; do
    python3 train_ppo.py \
      --prompts_fn=data/medium_sentences.txt \
      --model=./data/mix22.extended.txt.model2-final \
      --max_ppo_steps=100 \
      --init_kl_coef=0.5 \
      --target_kl=6.0 \
      --kl_horizon=${horizon} \
      --learning_rate=0.000015 \
      --use_score_scaling=True \
      --rule_reward_fac=${rule_reward_fac} \
      --suffix=reward-mix-${rule_reward_fac}-scaling-margin25.0-kl${kl}-horizon${horizon} \
      --reward_model=./data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.margin25.0-reward-model-final/ \
      | tee ./data/mix22.extended.txt.model2-final.${rule_reward_fac}-kl${kl}-horizon${horizon}.log.txt
  done
done
