import time
import random
import numpy as np

RAND_SEED = np.int(1e1*time.time()) % 4294967295
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)

n_rows, n_cols = 7,7
N_PLAYERS = 2

map_sz = (n_rows, n_cols)
map_szt = np.prod(map_sz)

n_input_channels = 3 # present and prior 2 game turns

########### training:
BATCH_SZ = 128
INPUTS_SHAPE = (BATCH_SZ, n_rows, n_cols, n_input_channels)

