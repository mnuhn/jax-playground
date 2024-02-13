function TRAIN() {
  WHAT="$1"
  DROPOUT="$2"
  echo
  echo
  echo "Running ${WHAT}"
  python3 train.py --data=./data/both.clean.small.10feature.1feature.256h.examples.npz --model=lstm --epochs=5 --nonconv_features=4 --dropout="${DROPOUT}" --tensorboard=true --draw=true --debug_every=5 --draw_every=20 --prefix=new --model_arch="${WHAT}"
}

# Recent timesteps
HIST_REC_32="I{fr:-32,to:0}"
HIST_REC_64="I{fr:-64,to:0}"
HIST_REC128="I{fr:-128,to:0}"

# Timesteps from 24h ago
HIST_24H_32="I{fr:-176,to:-112}"
HIST_24H_64="I{fr:-208,to:-112}"

# Summary of all 256 timesteps
HIST_ALL256="I{fr:-256,to:0}"
HIST_ALL128="I{fr:-256,to:0};M{w:2}"
HIST_ALL_64="I{fr:-256,to:0};M{w:4}"
HIST_ALL_32="I{fr:-256,to:0};M{w:8}"

# Convolutional layers.
CONV_16="C{k:8,ch:8};C{k:8,ch:16}"
CONV_32="C{k:8,ch:16};C{k:8,ch:32}"

# LSTM layers
LSTM_10="L{ch:10};L{ch:10}"
LSTM_30="L{ch:30};L{ch:30}"

# Maxpool layers
M_2="M{w:2}"
M_4="M{w:4}"

# Dense layers
DENSE_20="D{d:20};D{d:20}"
DENSE_40="D{d:40};D{d:40}"

# Stacks over all data points.
STACK_ALL256_FLAT_A="[${HIST_ALL256}]"
STACK_ALL256_FLAT_B="[${HIST_ALL256};${M_2}]"
STACK_ALL256_FLAT_C="[${HIST_ALL256};${M_4}]"

STACK_ALL256_A="[${HIST_ALL256};${CONV_16};${M_4};${LSTM_10}]"
STACK_ALL256_B="[${HIST_ALL256};${CONV_16};${M_4};${LSTM_30}]"
STACK_ALL256_C="[${HIST_ALL256};${CONV_32};${M_4};${LSTM_10}]"
STACK_ALL256_D="[${HIST_ALL256};${CONV_32};${M_4};${LSTM_30}]"

STACK_ALL256_E="[${HIST_ALL256};${CONV_16};${M_2};${LSTM_10}]"
STACK_ALL256_F="[${HIST_ALL256};${CONV_16};${M_2};${LSTM_30}]"
STACK_ALL256_G="[${HIST_ALL256};${CONV_32};${M_2};${LSTM_10}]"
STACK_ALL256_H="[${HIST_ALL256};${CONV_32};${M_2};${LSTM_30}]"


STACK_ALL_64_A="[${HIST_ALL_64};${CONV_16};${LSTM_10}]"
STACK_ALL_64_B="[${HIST_ALL_64};${CONV_16};${LSTM_30}]"
STACK_ALL_64_C="[${HIST_ALL_64};${CONV_32};${LSTM_10}]"
STACK_ALL_64_D="[${HIST_ALL_64};${CONV_32};${LSTM_30}]"

# Stacks over recent data points.
STACK_REC_32_FLAT_A="[${HIST_REC_32}]"
STACK_REC_32_FLAT_B="[${HIST_REC_32};${M_2}]"

STACK_REC_32_A="[${HIST_REC_32};${CONV_16};${LSTM_10}]"
STACK_REC_32_B="[${HIST_REC_32};${CONV_16};${LSTM_30}]"
STACK_REC_32_C="[${HIST_REC_32};${CONV_32};${LSTM_10}]"
STACK_REC_32_D="[${HIST_REC_32};${CONV_32};${LSTM_30}]"

STACK_REC_64_FLAT_A="[${HIST_REC_64}]"
STACK_REC_64_FLAT_B="[${HIST_REC_64};${M_2}]"
STACK_REC_64_FLAT_C="[${HIST_REC_64};${M_4}]"

STACK_REC_64_A="[${HIST_REC_64};${CONV_16};${LSTM_10}]"
STACK_REC_64_B="[${HIST_REC_64};${CONV_16};${LSTM_30}]"
STACK_REC_64_C="[${HIST_REC_64};${CONV_32};${LSTM_10}]"
STACK_REC_64_D="[${HIST_REC_64};${CONV_32};${LSTM_30}]"

STACK_REC128_FLAT_A="[${HIST_REC128}]"
STACK_REC128_FLAT_B="[${HIST_REC128};${M_2}]"
STACK_REC128_FLAT_C="[${HIST_REC128};${M_4}]"

STACK_REC128_A="[${HIST_REC128};${CONV_16};${LSTM_10}]"
STACK_REC128_B="[${HIST_REC128};${CONV_16};${LSTM_30}]"
STACK_REC128_C="[${HIST_REC128};${CONV_32};${LSTM_10}]"
STACK_REC128_D="[${HIST_REC128};${CONV_32};${LSTM_30}]"

# Stacks over data points around 24h ago.
STACK_24H_32_FLAT_A="[${HIST_24H_32}]"
STACK_24H_32_FLAT_B="[${HIST_24H_32};${M_2}}]"

STACK_24H_32_A="[${HIST_24H_32};${CONV_16};${LSTM_10}]"
STACK_24H_32_B="[${HIST_24H_32};${CONV_16};${LSTM_30}]"
STACK_24H_32_C="[${HIST_24H_32};${CONV_32};${LSTM_10}]"
STACK_24H_32_D="[${HIST_24H_32};${CONV_32};${LSTM_30}]"

STACK_24H_64_FLAT_A="[${HIST_24H_64}]"
STACK_24H_64_FLAT_B="[${HIST_24H_64};${M_2}]"

STACK_24H_64_A="[${HIST_24H_64};${CONV_16};${LSTM_10}]"
STACK_24H_64_B="[${HIST_24H_64};${CONV_16};${LSTM_30}]"
STACK_24H_64_C="[${HIST_24H_64};${CONV_32};${LSTM_10}]"
STACK_24H_64_D="[${HIST_24H_64};${CONV_32};${LSTM_30}]"

# Try out some.
TRAIN "[${STACK_ALL256_FLAT_A}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_ALL256_FLAT_B}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_ALL256_FLAT_A}];[${DENSE_20}]" 0.00

TRAIN "[${STACK_REC_32_FLAT_A}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_REC_32_FLAT_B}];[${DENSE_20}]" 0.00

TRAIN "[${STACK_REC_64_FLAT_A}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_REC_64_FLAT_B}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_REC_64_FLAT_C}];[${DENSE_20}]" 0.00

TRAIN "[${STACK_REC128_FLAT_A}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_REC128_FLAT_B}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_REC128_FLAT_C}];[${DENSE_20}]" 0.00

TRAIN "[${STACK_24H_32_FLAT_A}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_24H_32_FLAT_B}];[${DENSE_20}]" 0.00

exit 1

TRAIN "[${STACK_ALL_64_A}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_ALL_64_A}];[${DENSE_40}]" 0.00
TRAIN "[${STACK_ALL_64_B}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_ALL_64_B}];[${DENSE_40}]" 0.00
TRAIN "[${STACK_ALL_64_C}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_ALL_64_C}];[${DENSE_40}]" 0.00
TRAIN "[${STACK_ALL_64_D}];[${DENSE_20}]" 0.00
TRAIN "[${STACK_ALL_64_D}];[${DENSE_40}]" 0.00
