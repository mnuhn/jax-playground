Train Split: Wind data from <= 2022 (timeseries.lt2023.npy)
Test Split: Wind data from only 2023 (timeseries.2023.npy)

Results with CNNs:
batch_size 32 channels 20 history 257 conv_len 16 iters 49999 test_loss 0.3693382
batch_size 32 channels 20 history 257 conv_len 16 iters 28000 test_loss 0.3609284 # wind,gust,dir
batch_size 32 channels 20 history 257 conv_len 16 iters 60000 test_loss 0.36132392 #wind,gust,dir,airpressure
batch_size 64 channels 10 history 65 conv_len 8 iters 102000 test_loss 0.34981236 #wind,gust,dir,airpressure

all2.txt: wind gust dir airpressure
