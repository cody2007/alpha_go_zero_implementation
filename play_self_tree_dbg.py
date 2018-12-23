import tensorflow as tf
import numpy as np
import time
import os
import copy
import random
import global_vars as gv
from scipy.stats import pearsonr
from colorama import Fore, Style
from datetime import datetime
import architectures.tree_tf_op as arch
import py_util.py_util as pu
from scipy.special import gamma
import gnu_go_test as gt
sdir = 'models/' # directory to save and load models

save_nm = 'go_cpu_tree_0.200000EPS_7GMSZ_1000N_SIM_0.001000L2_LAMBDA_0.900000MOMENTUM_0.025000VAL_LAMBDA_1.000000CPUCT_20N_TURNS_128N_FILTERS_EPS0.110000_EPS0.020000_EPS0.010000.npy'

###### variables to save
save_vars = ['LSQ_LAMBDA', 'LSQ_REG_LAMBDA', 'POL_CROSS_ENTROP_LAMBDA', 'VAL_LAMBDA', 'VALR_LAMBDA', 'L2_LAMBDA',
	'FILTER_SZS', 'STRIDES', 'N_FILTERS', 'N_FC1', 'EPS', 'MOMENTUM', 'SAVE_FREQ', 'N_SIM', 'N_TURNS', 'CPUCT', 
	'N_EVAL_NN_GMS', 'N_EVAL_NN_GNU_GMS', 'N_EVAL_TREE_GMS', 'N_EVAL_TREE_GNU_GMS', 'CHKP_FREQ',
	'save_nm', 'DIR_A', 'start_time', 'EVAL_FREQ', 'boards', 'scores']
logs = ['val_mean_sq_err', 'pol_cross_entrop', 'pol_max_pre', 'pol_max', 'val_pearsonr','opt_batch','eval_batch']
print_logs = ['val_mean_sq_err', 'pol_cross_entrop', 'pol_max', 'val_pearsonr']

for nm in ['tree', 'nn']:
	for suffix in ['', '_gnu']:
		for key in ['win', 'n_captures', 'n_captures_opp', 'score', 'n_mvs', 'boards']:
			logs += ['%s_%s%s' % (key, nm, suffix)]

state_vars = ['log', 'run_time', 'global_batch', 'global_batch_saved', 'global_batch_evald', 'save_counter','boards', 'save_t'] # updated each save


save_d = np.load(sdir + save_nm).item()

for key in save_vars + state_vars:
	if key == 'save_nm':
		continue
	exec('%s = save_d["%s"]' % (key,key))

N_SIM = 400
N_TURNS = 40

########################################################################################

def ret_d(player): # return dictionary for input into tensorflow
	return {arch.moving_player: player, arch.dir_a: DIR_A, arch.dir_pre: dir_pre}

def session_backup(): # backup where we are in the tree (so we can do simulations and then return)
	arch.sess.run(arch.session_backup)
	pu.session_backup()

def session_restore(): # restore tree state
	arch.sess.run(arch.session_restore)
	pu.session_restore()

def run_sim(turn): # simulate game forward
	arch.sess.run(arch.session_backup)
	pu.session_backup()

	for sim in range(N_SIM):
		# backup then make next move
		for turn_sim in range(turn, N_TURNS+1):
			for player in [0,1]:
				# get valid moves, network policy and value estimates:
				valid_mv_map, pol, val = arch.sess.run([arch.valid_mv_map, arch.pol, arch.val], feed_dict=ret_d(player))

				# backup visit Q values
				if turn_sim != turn:
					pu.backup_visit(player, val)

				pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
				to_coords = pu.choose_moves(player, pol, CPUCT)[0] # choose moves based on policy and Q values (latter of which already stored in tree)
				pu.register_mv(player, to_coords) # register move in tree

				arch.sess.run(arch.move_frm_inputs, feed_dict={arch.moving_player: player, arch.to_coords_input: to_coords}) # move network (update GPU vars)
		
		# backup terminal state
		for player in [0,1]:
			winner = arch.sess.run(arch.winner, feed_dict=ret_d(player))
			pu.backup_visit(player, winner)
		
		# return move back to previous node in tree
		arch.sess.run(arch.session_restore)
		pu.session_restore()


############# init / load model
arch.init_model(N_FILTERS, FILTER_SZS, STRIDES, N_FC1, EPS, MOMENTUM,
	LSQ_LAMBDA, LSQ_REG_LAMBDA, POL_CROSS_ENTROP_LAMBDA, VAL_LAMBDA, VALR_LAMBDA, L2_LAMBDA)

arch.saver.restore(arch.sess, sdir + save_nm)

############### variable initialization:
dir_pre = 0

t_start = datetime.now()
run_time = datetime.now() - datetime.now()

vals = np.zeros((N_TURNS, 2, gv.BATCH_SZ), dtype='single')

######### generate batches
buffer_loc = 0

arch.sess.run(arch.init_state)
pu.init_tree()
turn_start_t = time.time()
for turn in range(N_TURNS):
	run_sim(turn)
	
	### make move
	for player in [0,1]:
		valid_mv_map, pol, vals[turn, player] = arch.sess.run([arch.valid_mv_map, arch.pol, arch.val], feed_dict = ret_d(player)) # generate batch and valid moves
		
		#########
		pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
		visit_count_map = pu.choose_moves(player, pol, CPUCT)[-1] # get number of times each node was visited
		
		to_coords = arch.sess.run([arch.tree_prob_visit_coord, arch.tree_prob_move_unit], feed_dict={arch.moving_player: player, 
			arch.visit_count_map: visit_count_map, arch.dir_pre: dir_pre, arch.dir_a: DIR_A})[0] # make move in proportion to visit counts

		pu.register_mv(player, to_coords) # register move in tree

		###############

	pu.prune_tree()
	
	if (turn+1) % 2 == 0:
		print 'finished turn', turn, time.time() - turn_start_t
		

winner = np.zeros((2, gv.BATCH_SZ))
for player in [0,1]:
	winner[player] = arch.sess.run(arch.winner, feed_dict=ret_d(player))

np.save('/tmp/vals_tree.npy', {'vals': vals, 'winner': winner})

