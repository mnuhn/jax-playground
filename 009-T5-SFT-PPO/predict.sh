
python3 predict_db.py \
  --prompts=data/medium_sentences.100.txt \
  --reward_model=./data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.reward-model-final \
  --skip_file=./data/mix22.txt \
  --max_len=60 \
  --model=./data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final \
  > ./data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.predictions.txt

exit 0

for W in 0.0 0.01 0.1 0.3 0.5; do
  python3 predict_db.py \
    --prompts=data/medium_sentences.100.txt \
    --reward_model=./data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.reward-model-final \
    --skip_file=./data/mix22.txt \
    --max_len=60 \
    --model=./data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.reward-mix-${W}.ppo-final \
    > ./data/mix22.extended.txt.model2-final.8antonyms-lenthwiggle.ppo-final.reward-mix-${W}.ppo-final.predictions.txt
done

