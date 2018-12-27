# kernprof -l bp_tree.py
# python -m line_profiler bp_tree.py.lprof  > p.py

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

################################### configuration: 
#### load previous model or start from scratch?
save_nm = None # this results in the optimization starting from scratch (comment out line below)

###### variables to save
save_vars = ['LSQ_LAMBDA', 'LSQ_REG_LAMBDA', 'POL_CROSS_ENTROP_LAMBDA', 'VAL_LAMBDA', 'VALR_LAMBDA', 'L2_LAMBDA',
	'FILTER_SZS', 'STRIDES', 'N_FILTERS', 'N_FC1', 'EPS', 'MOMENTUM', 'SAVE_FREQ', 'N_SIM', 'N_TURNS', 'CPUCT', 'N_TURNS_FRAC_TRAIN',
	'N_EVAL_NN_GMS', 'N_EVAL_NN_GNU_GMS', 'N_EVAL_TREE_GMS', 'N_EVAL_TREE_GNU_GMS', 'CHKP_FREQ', 'N_BATCH_SETS',
	'save_nm', 'DIR_A', 'start_time', 'EVAL_FREQ', 'boards', 'scores']
training_ex_vars = ['board', 'winner', 'tree_probs']

logs = ['val_mean_sq_err', 'pol_cross_entrop', 'pol_max_pre', 'pol_max', 'val_pearsonr','opt_batch','eval_batch']
print_logs = ['val_mean_sq_err', 'pol_cross_entrop', 'pol_max', 'val_pearsonr']

for nm in ['tree', 'nn']:
	for suffix in ['', '_gnu']:
		for key in ['win', 'n_captures', 'n_captures_opp', 'score', 'n_mvs', 'boards']:
			logs += ['%s_%s%s' % (key, nm, suffix)]

state_vars = ['log', 'run_time', 'global_batch', 'global_batch_saved', 'global_batch_evald', 'save_counter','boards', 'save_t'] # updated each save


################### start from scratch
restore = False
if save_nm is None:
	##### weightings on individual loss terms:
	LSQ_LAMBDA = 0
	LSQ_REG_LAMBDA = 0
	POL_CROSS_ENTROP_LAMBDA = 1
	VAL_LAMBDA = .025 #.05
	VALR_LAMBDA = 0
	L2_LAMBDA = 1e-3 # weight regularization 
	DIR_A = 0
	CPUCT = 1
	N_BATCH_SETS = 2

	##### model parameters
	N_LAYERS = 5 # number of model layers
	FILTER_SZS = [3]*N_LAYERS
	STRIDES = [1]*N_LAYERS
	F = 128 # number of filters
	N_FILTERS = [F]*N_LAYERS
	N_FC1 = 128 # number of units in fully connected layer
	
	
	EPS = 2e-1 # backprop step size
	MOMENTUM = .9

	N_SIM = 100 #200#5#10 # number of simulations at each turn
	N_TURNS = 40 # number of moves per player per game

	N_TURNS_FRAC_TRAIN = 1 #.5 # fraction of (random) turns to run bp on, remainder are discarded

	##### number of batch evaluations for testing model
	N_EVAL_NN_GMS = 1 # model evaluation for printing
	N_EVAL_NN_GNU_GMS = 1
	N_EVAL_TREE_GMS = 0 # model eval
	N_EVAL_TREE_GNU_GMS = 0

	######### save and checkpoint frequency
	SAVE_FREQ = N_TURNS*N_TURNS_FRAC_TRAIN
	EVAL_FREQ = SAVE_FREQ*1
	CHKP_FREQ = 60*60*10*2

	start_time = datetime.now()
	save_t = datetime.now()

	save_nm = 'go_%1.4fEPS_%iGMSZ_%iN_SIM_%iN_TURNS_%iN_FILTERS_%iN_LAYERS_%iN_BATCH_SETS.npy' % (EPS, gv.n_rows, N_SIM, N_TURNS, N_FILTERS[0], N_LAYERS, N_BATCH_SETS)

	boards = {}; scores = {} # eval
	save_d = {}
	for key in save_vars:
		exec('save_d["%s"] = %s' % (key,key))
	save_d['script_nm'] = __file__

	global_batch = 0
	global_batch_saved = 0
	global_batch_evald = 0
	save_counter = 0

	log = {}
	for key in logs:
		log[key] = []

else:
	restore = True
	save_d = np.load(sdir + save_nm).item()

	for key in save_vars + state_vars:
		if key == 'save_nm':
			continue
		exec('%s = save_d["%s"]' % (key,key))

	EPS_ORIG = EPS
	#EPS = 2e-3
	#EPS = 5e-3

	save_nm_orig = save_nm

	# append new learning rate to file name if not already added
	append_txt = '_EPS%f.npy' % EPS
	if EPS_ORIG != EPS and save_nm.find(append_txt) == -1:
		save_nm = save_nm.split('.npy')[0] + append_txt
	print 'saving to:'
	print save_nm

########################################################################################

#@profile
def sv(): # save game
	#return

	global save_d, save_t
	# update state vars
	for key in state_vars + training_ex_vars:
		exec('save_d["%s"] = %s' % (key, key))

	# save
	save_nms = [save_nm]
	if (datetime.now() - save_t).seconds > CHKP_FREQ:
		save_nms += [save_nm + str(datetime.now())]
		save_t = datetime.now()
	
	for nm in save_nms:
		np.save(sdir + nm, save_d)
		arch.saver.save(arch.sess, sdir + nm)

#@profile
def ret_d(player): # return dictionary for input into tensorflow
	return {arch.moving_player: player, arch.dir_a: DIR_A, arch.dir_pre: dir_pre}

def session_backup(): # backup where we are in the tree (so we can do simulations and then return)
	arch.sess.run(arch.session_backup)
	pu.session_backup()

def session_restore(): # restore tree state
	arch.sess.run(arch.session_restore)
	pu.session_restore()

#@profile
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

if restore:
	print 'restore nm %s' % save_nm_orig
	arch.saver.restore(arch.sess, sdir + save_nm_orig)


############# tensorflow operations to evaluate at each backprop step
bp_eval_nodes = [arch.train_step, arch.val_mean_sq_err, arch.pol_cross_entrop_err, arch.val_pearsonr]

############### variable initialization:
#if DIR_A == 0:
dir_pre = 0
#else:
#	dir_pre = gamma(DIR_A * gv.map_szt) / (gamma(DIR_A)**gv.map_szt)

BUFFER_SZ = N_BATCH_SETS * N_TURNS * 2 * gv.BATCH_SZ
board = np.zeros((BUFFER_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels),  dtype='single')
winner = np.zeros((N_BATCH_SETS, N_TURNS, 2, gv.BATCH_SZ), dtype='single')
tree_probs = np.zeros((N_BATCH_SETS, BUFFER_SZ/N_BATCH_SETS, gv.map_szt+1), dtype='single')

inds_total = np.arange(BUFFER_SZ)

err_denom = 0
val_mean_sq_err = 0; pol_cross_entrop_err = 0; val_pearsonr = 0

buffer_loc = 0
batch_set = 0
batch_sets_created = 0
t_start = datetime.now()
run_time = datetime.now() - datetime.now()

# save
print '------------- saving to ', save_nm
sv()


######### generate batches
if buffer_loc >= BUFFER_SZ:
	buffer_loc = 0
	batch_set = 0

arch.sess.run(arch.init_state)
pu.init_tree()
turn_start_t = time.time()
for turn in range(N_TURNS):
	run_sim(turn)
	
	### make move
	for player in [0,1]:
		inds = buffer_loc + np.arange(gv.BATCH_SZ)
		board[inds], valid_mv_map, pol = arch.sess.run([arch.imgs, arch.valid_mv_map, arch.pol], feed_dict = ret_d(player)) # generate batch and valid moves
		
		#########
		pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
		visit_count_map = pu.choose_moves(player, pol, CPUCT)[-1] # get number of times each node was visited
		
		to_coords = arch.sess.run([arch.tree_prob_visit_coord, arch.tree_prob_move_unit], feed_dict={arch.moving_player: player, 
			arch.visit_count_map: visit_count_map, arch.dir_pre: dir_pre, arch.dir_a: DIR_A})[0] # make move in proportion to visit counts

		pu.register_mv(player, to_coords) # register move in tree
		###############
	
		assert False

		buffer_loc += gv.BATCH_SZ
	if turn == 0:
		assert False
	pu.prune_tree()
	
	if (turn+1) % 2 == 0:
		print 'finished turn %i, %i secs (%i)' % (turn, time.time() - turn_start_t, batch_sets_created)
		
##### create prob maps
for player in [0,1]:
	winner[batch_set, :, player] = arch.sess.run(arch.winner, feed_dict={arch.moving_player: player})
tree_probs[batch_set] = pu.return_probs_map(N_TURNS)


