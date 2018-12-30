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
	'FILTER_SZS', 'STRIDES', 'N_FILTERS', 'N_FC1', 'EPS', 'MOMENTUM', 'SAVE_FREQ', 'N_SIM', 'N_TURNS', 'CPUCT', 
	'N_EVAL_NN_GMS', 'N_EVAL_NN_GNU_GMS', 'N_EVAL_TREE_GMS', 'N_EVAL_TREE_GNU_GMS', 'CHKP_FREQ', 'BUFFER_SZ', 'N_BATCH_SETS',
	'save_nm', 'DIR_A', 'start_time', 'EVAL_FREQ', 'boards', 'scores']

training_ex_vars = ['board', 'winner', 'tree_probs', 'batch_set', 'batch_sets_created', 'buffer_loc']

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
	N_BATCH_SETS = 2 # number of batch sets to store in training buffer

	batch_set = 0
	batch_sets_created = 0
	buffer_loc = 0

	##### model parameters
	N_LAYERS = 5 # number of model layers
	FILTER_SZS = [3]*N_LAYERS
	STRIDES = [1]*N_LAYERS
	F = 128 # number of filters
	N_FILTERS = [F]*N_LAYERS
	N_FC1 = 128 # number of units in fully connected layer
	
	
	EPS = 2e-1 # backprop step size
	MOMENTUM = .9

	N_SIM = 300 # number of simulations at each turn
	N_TURNS = 35 # number of moves per player per game

	#### training buffers
	BUFFER_SZ = N_BATCH_SETS * N_TURNS * 2 * gv.BATCH_SZ

	board = np.zeros((BUFFER_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels),  dtype='single')
	winner = np.zeros((N_BATCH_SETS, N_TURNS, 2, gv.BATCH_SZ), dtype='single')
	tree_probs = np.zeros((N_BATCH_SETS, BUFFER_SZ/N_BATCH_SETS, gv.map_szt), dtype='single')

	##### number of batch evaluations for testing model
	N_EVAL_NN_GMS = 1 # model evaluation for printing
	N_EVAL_NN_GNU_GMS = 1
	N_EVAL_TREE_GMS = 0 # model eval
	N_EVAL_TREE_GNU_GMS = 0

	######### save and checkpoint frequency
	SAVE_FREQ = N_TURNS
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

	for key in save_vars + state_vars + training_ex_vars:
		if key == 'save_nm':
			continue
		exec('%s = save_d["%s"]' % (key,key))

	EPS_ORIG = EPS
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

inds_total = np.arange(BUFFER_SZ)

err_denom = 0
val_mean_sq_err = 0; pol_cross_entrop_err = 0; val_pearsonr = 0

buffer_loc = 0
init_buffers = 0
t_start = datetime.now()
run_time = datetime.now() - datetime.now()

# save
print '------------- saving to ', save_nm
sv()


######################################### training loop:
while True:
	######### generate batches
	while batch_sets_created < N_BATCH_SETS:
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
				
				buffer_loc += gv.BATCH_SZ

			pu.prune_tree()
			
			if (turn+1) % 2 == 0:
				print 'finished turn', turn, time.time() - turn_start_t
				

		##### create prob maps
		for player in [0,1]:
			winner[batch_set, :, player] = arch.sess.run(arch.winner, feed_dict={arch.moving_player: player})
		tree_probs[batch_set] = pu.return_probs_map(N_TURNS)

		batch_set += 1
		batch_sets_created += 1

	batch_sets_created -= 1

	#############################
	# train
	random.shuffle(inds_total)
	for batch in range(N_TURNS):
		inds = inds_total[batch*gv.BATCH_SZ + np.arange(gv.BATCH_SZ)]
		
		board2, tree_probs2 = pu.rotate_reflect_imgs(board[inds], tree_probs.reshape((BUFFER_SZ, gv.map_szt))[inds]) # rotate and reflect board randomly

		train_dict = {arch.imgs: board2,
				arch.pol_target: tree_probs2,
				arch.val_target: winner.ravel()[inds]}

		val_mean_sq_err_tmp, pol_cross_entrop_err_tmp, val_pearsonr_tmp = arch.sess.run(bp_eval_nodes, feed_dict=train_dict)[1:]

		# update logs
		val_mean_sq_err += val_mean_sq_err_tmp
		pol_cross_entrop_err += pol_cross_entrop_err_tmp
		val_pearsonr += val_pearsonr_tmp
		global_batch += 1
		err_denom += 1

	
	############################
	# save, evaluate network, print stats

	if (global_batch - global_batch_saved) >= SAVE_FREQ or global_batch == N_TURNS:
		global_batch_saved = global_batch
		
		##### network evaluation against random player and GNU Go
		if (global_batch - global_batch_evald) >= EVAL_FREQ:
			global_batch_evald = global_batch
			t_eval = time.time()
			print 'evaluating nn'

			d = ret_d(0)

			for nm, N_GMS_L in zip(['nn','tree'], [[N_EVAL_NN_GNU_GMS, N_EVAL_NN_GMS], [N_EVAL_TREE_GMS, N_EVAL_TREE_GNU_GMS]]):
				for gnu, N_GMS in zip([True,False], N_GMS_L):
					key = '%s%s' % (nm, '' + gnu*'_gnu')
					t_key = time.time()
					boards[key] = np.zeros((N_TURNS,) + gv.INPUTS_SHAPE[:-1], dtype='int8')
					n_mvs = 0.; win_eval = 0.; score_eval = 0.; n_captures_eval = np.zeros(2, dtype='single')
					for gm in range(N_GMS):
						arch.sess.run(arch.init_state)
						pu.init_tree()
						# init gnu state
						if gnu:
							gt.init_board(arch.sess.run(arch.gm_vars['board']))

						for turn in range(N_TURNS):
							board_tmp = arch.sess.run(arch.gm_vars['board'])
						
							#### search / make move
							if nm == 'tree':
								run_sim(turn)
								assert False
							else:
								# prob choose first move, deterministically choose remainder
								if turn == 0:
									to_coords = arch.sess.run([arch.nn_prob_to_coords_valid_mvs, arch.nn_prob_move_unit_valid_mvs], feed_dict=d)[0]
								else:
									to_coords = arch.sess.run([arch.nn_max_prob_to_coords_valid_mvs, arch.nn_max_prob_move_unit_valid_mvs], feed_dict=d)[0]


							board_tmp2 = arch.sess.run(arch.gm_vars['board'])
							n_mvs += board_tmp.sum() - board_tmp2.sum()

							# move opposing player
							if gnu:
								gt.move_nn(to_coords) 

								# mv gnugo
								ai_to_coords = gt.move_ai()
								arch.sess.run(arch.imgs, feed_dict={arch.moving_player: 1})
								arch.sess.run(arch.nn_max_move_unit, feed_dict={arch.moving_player: 1, arch.nn_max_to_coords: ai_to_coords})
							else:
								arch.sess.run(arch.imgs, feed_dict = ret_d(1))
								arch.sess.run(arch.move_random_ai, feed_dict = ret_d(1))
		
							boards[key][turn] = arch.sess.run(arch.gm_vars['board'])

							if nm == 'tree':
								pu.prune_tree()
							# turn

						# save stats
						win_tmp, score_tmp, n_captures_tmp = arch.sess.run([arch.winner, arch.score, arch.n_captures], feed_dict={arch.moving_player: 0})
						scores[key] = copy.deepcopy(score_tmp)

						win_eval += win_tmp.mean()
						score_eval += score_tmp.mean()
						n_captures_eval += n_captures_tmp.mean(1)
						# gm
					
					# log
					log['win_' + key].append( (win_eval / (2*np.single(N_GMS))) + .5 )
					log['n_captures_' + key].append( n_captures_eval[0] / np.single(N_GMS) )
					log['n_captures_opp_' + key].append( n_captures_eval[1] / np.single(N_GMS) )
					log['score_' + key].append( score_eval / np.single(N_GMS) )
					log['n_mvs_' + key].append( n_mvs / np.single(N_GMS * N_TURNS * gv.BATCH_SZ) )

					log['boards_' + key].append( boards[key][-1] )
					print key, 'eval time', time.time() - t_key
					# gnu
				# nm
			log['eval_batch'].append( global_batch )
			print 'eval time', time.time() - t_eval
			# eval
		####################### end network evaluation

		pol, pol_pre = arch.sess.run([arch.pol, arch.pol_pre], feed_dict={arch.moving_player: 0})

		##### log
		log['val_mean_sq_err'].append ( val_mean_sq_err / err_denom )
		log['pol_cross_entrop'].append( pol_cross_entrop_err / err_denom )
		log['val_pearsonr'].append( val_pearsonr / err_denom )
		log['opt_batch'].append( global_batch )

		log['pol_max_pre'].append( np.median(pol_pre.max(1)) )
		log['pol_max'].append( np.median(pol.max(1)) )

		val_mean_sq_err = 0
		pol_cross_entrop_err = 0
		val_pearsonr = 0
		err_denom = 0
		
		# save
		sv()

		########## print
		run_time += datetime.now() - t_start

		if (save_counter % 20) == 0:
			print
			print Style.BRIGHT + Fore.GREEN + save_nm, Fore.WHITE + 'EPS', EPS, 'start', str(start_time).split('.')[0], 'run time', \
					str(run_time).split('.')[0]
			print
		save_counter += 1

		print_str = '%i' % global_batch
		for key in print_logs:
			print_str += ' %s ' % key
			if isinstance(log[key], int):
				print_str += str(log[key][-1])
			else:
				print_str += '%1.4f' % log[key][-1]

		print_str += ' %4.1f' % (datetime.now() - t_start).total_seconds()
		print print_str
		
		t_start = datetime.now()


