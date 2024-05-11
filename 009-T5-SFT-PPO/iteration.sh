TRAINING_EXAMPLES_FN=$1
MODEL_FN=${TRAINING_EXAMPLES_FN}.model2
MODEL_FINAL_FN=${TRAINING_EXAMPLES_FN}.model2-final
DB_FN=${TRAINING_EXAMPLES_FN}.predictions2.db
PROMPTS_FN=data/medium_sentences.5000.txt

if [ ! -f "${TRAINING_EXAMPLES_FN}" ]; then
  echo "File '${TRAINING_EXAMPLES_FN}' does not exist."
  exit 1
fi

# check files do not exist
#assert False

python3 train_sft.py \
  --training_data="${TRAINING_EXAMPLES_FN}" \
  --model_out="${MODEL_FN}"

python predict_db.py \
  --db=${DB_FN} \
  --model=${MODEL_FINAL_FN} \
  --prompts=${PROMPTS_FN} \
  --skip_file=${TRAINING_EXAMPLES_FN} \
  --num_per_prompt=100 \
  --max_len=60

python3 viewer_sft.py \
  --db=${DB_FN}

#python3 get_completions_from_db.py --db="${DB_FN}" --rated_only=True > "${TRAINING_EXAMPLES_FN}.handpicked.txt" # data/mix20.txt.predictions.db.handpicked.txt
