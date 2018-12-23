import copy
import os
import numpy as np
from numpy import sqrt
import time
import global_vars as gv
import tensorflow as tf
import architectures.tree_tf_op as arch
import py_util.py_util as pu

########################################################## configuration:
save_nm = 'models/go_cpu_tree_0.200000EPS_7GMSZ_1000N_SIM_0.001000L2_LAMBDA_0.900000MOMENTUM_0.025000VAL_LAMBDA_1.000000CPUCT_20N_TURNS_128N_FILTERS_EPS0.110000_EPS0.020000_EPS0.010000.npy'

# load the following variables from the model .npy file:
save_vars = ['LSQ_LAMBDA', 'LSQ_REG_LAMBDA', 'POL_CROSS_ENTROP_LAMBDA', 'VAL_LAMBDA', 'VALR_LAMBDA', 'L2_LAMBDA',
	'FILTER_SZS', 'STRIDES', 'N_FILTERS', 'N_FC1', 'EPS', 'MOMENTUM', 'SAVE_FREQ', 'N_SIM', 'N_TURNS', 'CPUCT', 'DIR_A']
save_d = np.load(save_nm).item()
for key in save_vars:
	if key == 'save_nm':
		continue
	exec('%s = save_d["%s"]' % (key,key))

########## over-write number of simulations previously used:
CPUCT = 1
if DIR_A == 0:
	dir_pre = 0
else:
	dir_pre = gamma(DIR_A * gv.map_szt) / (gamma(DIR_A)**gv.map_szt)

###############################################################################

############## load model, init variables
arch.init_model(N_FILTERS, FILTER_SZS, STRIDES, N_FC1, EPS, MOMENTUM,
	LSQ_LAMBDA, LSQ_REG_LAMBDA, POL_CROSS_ENTROP_LAMBDA, VAL_LAMBDA, VALR_LAMBDA, L2_LAMBDA)

arch.saver.restore(arch.sess, save_nm)
arch.sess.run(arch.init_state)
pu.init_tree()

N_TURNS = 40

def ret_d(player): # return dictionary for input into tensor flow
	return {arch.moving_player: player, arch.dir_a: DIR_A, arch.dir_pre: dir_pre}

# move neural network
t_start = time.time()
arch.sess.run(arch.session_backup)
pu.init_tree()
pu.session_backup()

vals = np.zeros((N_TURNS, 2, gv.BATCH_SZ), dtype='single')
boards = np.zeros((N_TURNS,) + gv.INPUTS_SHAPE[:-1], dtype='int8')

for turn in range(N_TURNS):
	for player in [0,1]:
		valid_mv_map, vals[turn, player] = arch.sess.run([arch.valid_mv_map, arch.val], feed_dict=ret_d(player)) ## dbg

		if turn == 0:	
			arch.sess.run(arch.nn_prob_move_unit_valid_mvs, feed_dict=ret_d(player))
		else:
			arch.sess.run(arch.nn_max_prob_move_unit_valid_mvs, feed_dict=ret_d(player))
	boards[turn] = arch.sess.run(arch.gm_vars['board'])

winner = np.zeros((2, gv.BATCH_SZ))
for player in [0,1]:
	winner[player] = arch.sess.run(arch.winner, feed_dict=ret_d(player))

np.save('/tmp/vals.npy', {'vals': vals, 'winner': winner, 'boards': boards})

