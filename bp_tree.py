# ------------
# model copies:
# ------------
# eval32: model to run bp on, the model which all others are eventually updated to
# eval: float16 versions of `eval32`. updated to follow backprop (the `eval32` model)
# main: older version of `eval` that `eval` model must win against with certainty p < .05
#       Once the benchmark is reached, `main` is updated to `eval32`.
#	`main` is used to create all training batches

import os.path
import pygame
import scipy.stats
import copy
import random
import multiprocessing as mp
import time
import numpy as np
import tensorflow as tf
import global_vars as gv
from datetime import datetime
import architectures.tree_tf_op_multi as arch # the tensorflow model definitions
import py_util.py_util as pu # operates and stores the move branching tree
import gnu_go_test as gt # playing against gnu go
from colorama import Fore, Style
sdir = 'models/' # directory to save and load models

################################### configuration: 
#### load previous model or start from scratch? (set save_nm = None if you want to start from scratch, i.e, create a new model)
#save_nm = None
save_nm = 'go_0.2000EPS_7GMSZ_800N_SIM_32N_TURNS_128N_FILTERS_5N_LAYERS_35N_BATCH_SETS_TOTAL_35_N_BATCH_SETS_MIN_5N_REP_TRAIN.npy'

if True: # run on two gpus
	MASTER_WORKER = 0
	GPU_LIST = [0,1] # gpu card ids
else: # run on one gpu only
	MASTER_WORKER = 1
	GPU_LIST = [1]

###### variables to save
save_vars = ['LSQ_LAMBDA', 'LSQ_REG_LAMBDA', 'POL_CROSS_ENTROP_LAMBDA', 'VAL_LAMBDA', 'VALR_LAMBDA', 'L2_LAMBDA', 'N_REP_TRAIN',
	'FILTER_SZS', 'STRIDES', 'N_FILTERS', 'N_FC1', 'EPS', 'MOMENTUM', 'SAVE_FREQ', 'N_SIM', 'N_TURNS', 'CPUCT', 
	'N_EVAL_NN_GMS', 'N_EVAL_NN_GNU_GMS', 'N_EVAL_TREE_GMS', 'N_EVAL_TREE_GNU_GMS', 'CHKP_FREQ', 'BUFFER_SZ', 'N_BATCH_SETS_MIN', 'N_BATCH_SETS_BLOCK', 'N_BATCH_SETS_TOTAL',
	'save_nm', 'start_time', 'EVAL_FREQ', 'boards', 'scores', 'GATE_THRESH', 'N_GATE_BATCH_SETS']

training_ex_vars = ['board', 'winner', 'tree_probs', 'batch_set', 'batch_sets_created', 'batch_sets_created_total', 'buffer_loc']

logs = ['val_mean_sq_err', 'pol_cross_entrop', 'pol_max_pre', 'pol_max', 'val_pearsonr','opt_batch','eval_batch',
	'self_eval_win_rate', 'model_promoted', 'self_eval_perc']
print_logs = ['val_mean_sq_err', 'pol_cross_entrop', 'pol_max', 'val_pearsonr']

for nm in ['tree', 'nn']:
	for suffix in ['', '_gnu']:
		for key in ['win', 'n_captures', 'n_captures_opp', 'score', 'n_mvs', 'boards']:
			logs += ['%s_%s%s' % (key, nm, suffix)]

state_vars = ['log', 'run_time', 'global_batch', 'global_batch_saved', 'global_batch_evald', 'save_counter','boards', 'save_t'] # updated each save

##########################################
def ret_d(player): # return dictionary for input into tensorflow
	return {arch.moving_player: player}

# simulate making moves (i.e., use the tree search)
# `scopes` controls which models to use (and their ordering of who plays first)
def run_sim(turn, starting_player, scopes=['main', 'main']):
	arch.sess.run(arch.session_backup)
	pu.session_backup()

	for sim in range(N_SIM):
		# backup then make next move
		for turn_sim in range(turn, N_TURNS+1):
			for player, s in zip([0,1], scopes):
				if turn_sim == turn and starting_player == 1 and player == 0: # skip player 0, has already moved
					continue

				# get valid moves, network policy and value estimates:
				valid_mv_map, pol, val = arch.sess.run([arch.valid_mv_map, arch.pol[s], arch.val[s]], feed_dict=ret_d(player))

				# backup visit Q values
				if turn_sim != turn:
					pu.backup_visit(player, np.array(val, dtype='single'))

				pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
				to_coords = pu.choose_moves(player, np.array(pol, dtype='single'), CPUCT)[0] # choose moves based on policy and Q values (latter of which already stored in tree)
				pu.register_mv(player, np.array(to_coords, dtype='int32')) # register move in tree

				arch.sess.run(arch.move_frm_inputs, feed_dict={arch.moving_player: player, arch.to_coords_input: to_coords}) # move network (update GPU vars)
		
		############ backup terminal state
		winner = np.array(arch.sess.run(arch.winner, feed_dict=ret_d(0)), dtype='single')

		# update tree with values (outcomes) of each game)
		pu.backup_visit(0, winner)
		pu.backup_visit(1, -winner)

		# return move back to previous node in tree
		arch.sess.run(arch.session_restore) # reset gpu game state
		pu.session_restore() # reset cpu tree state


####################################
shared_nms = ['buffer_loc', 'batch_sets_created', 'batch_sets_created_total', 'batch_set', 's_board', 's_winner', 's_tree_probs', 'weights_changed', 'buffer_lock', 'weights_lock', 'save_nm', 'new_model', 'weights', 'weights_eval',\
	'eval_games_won', 'eval_batch_sets_played', 'eval_stats_lock', 'scope_next', 'eval_batch_sets_main_first']
# ^ update sv() to handle shared variables

def init(i_buffer_loc, i_batch_sets_created, i_batch_sets_created_total, i_batch_set, i_s_board, i_s_winner, i_s_tree_probs, i_weights_changed, i_buffer_lock, i_weights_lock, i_save_nm, i_new_model, i_weights, i_weights_eval, i_eval_games_won, i_eval_batch_sets_played, i_eval_stats_lock, i_scope_next, i_eval_batch_sets_main_first):
	for nm in shared_nms:
		exec('global ' + nm)
		exec('%s = i_%s' % (nm, nm))

#####################################################################################################################
def worker_save_shapes(i):
	#### restore
	save_d = np.load(sdir + save_nm, allow_pickle=True).item()

	for key in save_vars + state_vars + training_ex_vars:
		if (key == 'save_nm') or (key in shared_nms):
			continue
		exec('%s = save_d["%s"]' % (key,key))

	############# init / load model
	DEVICE = '/gpu:%i' % i
	arch.init_model(DEVICE, N_FILTERS, FILTER_SZS, STRIDES, N_FC1, EPS, MOMENTUM,
		LSQ_LAMBDA, LSQ_REG_LAMBDA, POL_CROSS_ENTROP_LAMBDA, VAL_LAMBDA, VALR_LAMBDA, L2_LAMBDA)


	weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
	weights_l = []
	for k in range(len(weights)):
		weights_l.append(tuple(weights[k].shape.as_list()))
	
	return weights_l

# worker: sets weights from shared variables if they've been updated by the master worker
def set_weights():
	if WORKER_ID == MASTER_WORKER: # return if we are the master worker
		return False
	
	with weights_lock:
		if weights_changed.value == 0: # weights haven't been changed
			return False

		for i in range(len(weights_current)):
			# set `main` model copy from shared weights
			w = np.frombuffer(weights[i].get_obj(), 'float16')
			w = w.reshape(tuple(weights_current[i].shape.as_list()))
			weights_current[i].load(w)
			
			# set `eval` model copy from shared weights
			w = np.frombuffer(weights_eval[i].get_obj(), 'float16')
			w = w.reshape(tuple(weights_eval_current[i].shape.as_list()))
			weights_eval_current[i].load(w)

		weights_changed.value = 0
	return True

# master: set shared variables to values loaded from restore file
#         (the values were from the checkpoint into the tensorflow variables)
#	  (this is only done once ever -- once model training is started for 1st time)
def set_all_shared_to_loaded(): 	
	assert WORKER_ID == MASTER_WORKER # only the master worker should do this
	with weights_lock:
		weights_current_vals = arch.sess.run(weights_current) # `main` from tf
		weights_eval_current_vals = arch.sess.run(weights_eval_current)

		for i in range(len(weights_current)):
			# set `main` shared variables = `main` from tf
			w = np.frombuffer(weights[i].get_obj(), 'float16')
			w[:] = weights_current_vals[i].ravel()
			
			# set `eval` shared variables = `main` from tf
			w = np.frombuffer(weights_eval[i].get_obj(), 'float16')
			w[:] = weights_eval_current_vals[i].ravel()

# master: set shared variables `main` and `eval` (and tf vars) from current tensorflow copy of eval32
def set_all_to_eval32_and_get():
	assert WORKER_ID == MASTER_WORKER # only the master worker should do this
	with weights_lock:
		weights_eval32_current_vals = arch.sess.run(weights_eval32_current) # `eval32` from tf

		for i in range(len(weights_current)):
			# set `main` shared variables = `eval32` from tf
			w = np.frombuffer(weights[i].get_obj(), 'float16')
			w[:] = weights_eval32_current_vals[i].ravel()
			
			# set `eval` shared variables = `eval32` from tf
			w = np.frombuffer(weights_eval[i].get_obj(), 'float16')
			w[:] = weights_eval32_current_vals[i].ravel()

			# update tf copy
			weights_current[i].load(weights_eval32_current_vals[i]) # `main`
			weights_eval_current[i].load(weights_eval32_current_vals[i]) # `eval`
		
		weights_changed.value = 1

# master: update `eval` to values from backprop (current `eval32` tf weights)
def set_eval16_to_eval32_start_eval():
	assert WORKER_ID == MASTER_WORKER
	with weights_lock and eval_stats_lock:
		weights_eval32_current_vals = arch.sess.run(weights_eval32_current) # `eval32` from tf

		for i in range(len(weights_current)):
			# set `eval` shared variables = `eval32` from tf
			w = np.frombuffer(weights_eval[i].get_obj(), 'float16')
			w[:] = weights_eval32_current_vals[i].ravel()
			
			# update tensorflow `eval` model = `eval32`
			weights_eval_current[i].load(weights_eval32_current_vals[i])

		weights_changed.value = 1
		eval_games_won.value = 0
		eval_batch_sets_played.value = 0
		scope_next.value = 0
		eval_batch_sets_main_first.value = 0


def print_eval_stats():
	p_val = scipy.stats.binom_test(eval_games_won.value, eval_batch_sets_played.value*gv.BATCH_SZ, alternative='greater') 
	model_outperforms = p_val < .05
	perc = 100*np.single(eval_games_won.value)/(eval_batch_sets_played.value * gv.BATCH_SZ)
	pstr = 'eval wins %i' % eval_games_won.value
	pstr += ' sets played %i' % eval_batch_sets_played.value
	pstr += ' percent %1.2f' % perc
	pstr += ' p %1.3f' % p_val
	pstr += ' pass %i' % model_outperforms
	print pstr
	return model_outperforms, perc

# plays 2*N_GATE_BATCH_SETS rounds of batches, ensuring ordering of eval and main are balanced
# will also terminate at end of current batch eval if N_GATE_BATCH_SETS+1 have been played
# scope_next: alternates between 0,1 at start of each new batch set. to order which player goes first
def eval_model():
	set_weights()

	while True:
		arch.sess.run(arch.init_state)
		pu.init_tree()
		turn_start_t = time.time()
		
		### choose order
		with eval_stats_lock:
			if scope_next.value == 0:
				scopes = ['main', 'eval']
			else:
				scopes = ['eval', 'main']

			scope_next.value = 1 - scope_next.value
		
		scopes = np.asarray(scopes)
		
		for turn in range(N_TURNS):
			### make move
			for player, s in zip([0,1], scopes):
				if eval_batch_sets_played.value >= (2*N_GATE_BATCH_SETS):
					return # finished
							
				run_sim(turn, player, scopes=scopes)

				valid_mv_map, pol = arch.sess.run([arch.valid_mv_map, arch.pol[s]], feed_dict = ret_d(player)) # generate batch and valid moves
				
				#########
				pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
				visit_count_map = pu.choose_moves(player, np.array(pol, dtype='single'), CPUCT)[-1] # get number of times each node was visited
				
				to_coords = arch.sess.run([arch.tree_prob_visit_coord, arch.tree_prob_move_unit], feed_dict={arch.moving_player: player, 
					arch.visit_count_map: visit_count_map})[0] # make move in proportion to visit counts

				pu.register_mv(player, np.array(to_coords, dtype='int32')) # register move in tree

			pu.prune_tree(0)
			
			if (turn+1) % 2 == 0:
				print 'eval finished turn %i (%i sec) GPU %i eval_batch_sets_played %i' % (turn, time.time() - turn_start_t, WORKER_ID, eval_batch_sets_played.value)
				

		with eval_stats_lock:
			# do not add any more stats for these conditions
			if eval_batch_sets_main_first.value >= N_GATE_BATCH_SETS and scopes[0] == 'main':
				continue
			if (eval_batch_sets_played.value - eval_batch_sets_main_first.value) >= N_GATE_BATCH_SETS and scopes[0] == 'eval':
				continue

			eval_player = np.nonzero(scopes == 'eval')[0][0]
			res = arch.sess.run(arch.winner, feed_dict={arch.moving_player: eval_player})
			print 'ties', (res == 0).sum(), 'wins', (res == 1).sum(), 'rate %2.3f' % ((res == 1).sum()/np.single(gv.BATCH_SZ)), 'opp wins', (res == -1).sum(), scopes
			eval_games_won.value += np.int((res == 1).sum())
			eval_batch_sets_played.value += 1
			eval_batch_sets_main_first.value += int(scopes[0] == 'main')
			print_eval_stats()


def worker(i_WORKER_ID):
	global WORKER_ID, weights_current, weights_eval_current, weights_eval32_current, val_mean_sq_err, pol_cross_entrop_err, val_pearsonr
	global board, winner, tree_probs, save_d, bp_eval_nodes, t_start, run_time, save_nm
	WORKER_ID = i_WORKER_ID

	err_denom = 0; val_pearsonr = 0
	val_mean_sq_err = 0; pol_cross_entrop_err = 0; 
	t_start = datetime.now()
	run_time = datetime.now() - datetime.now()

	#### restore
	save_d = np.load(sdir + save_nm, allow_pickle=True).item()

	for key in save_vars + state_vars + training_ex_vars:
		if (key == 'save_nm') or (key in shared_nms):
			continue
		exec('global ' + key)
		exec('%s = save_d["%s"]' % (key,key))

	EPS_ORIG = EPS
	#EPS = 2e-3 ###################################################### < overrides previous backprop step sizes
	
	############# init / load model
	DEVICE = '/gpu:%i' % WORKER_ID
	arch.init_model(DEVICE, N_FILTERS, FILTER_SZS, STRIDES, N_FC1, EPS, MOMENTUM,
		LSQ_LAMBDA, LSQ_REG_LAMBDA, POL_CROSS_ENTROP_LAMBDA, VAL_LAMBDA, VALR_LAMBDA, L2_LAMBDA)

	bp_eval_nodes = [arch.train_step, arch.val_mean_sq_err, arch.pol_cross_entrop_err, arch.val_pearsonr]
	
	# ops for trainable weights
	weights_current = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
	weights_eval_current = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval/')
	weights_eval32_current = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval32')

	if new_model == False:
		print 'restore nm %s' % save_nm
		arch.saver.restore(arch.sess, sdir + save_nm)
		if WORKER_ID == MASTER_WORKER:
			set_all_shared_to_loaded()
	else: #### sync model weights
		if WORKER_ID == MASTER_WORKER:
			set_all_to_eval32_and_get()
		else:
			while set_weights() == False: # wait for weights to be set
				continue
	###### shared variables
	board = np.frombuffer(s_board.get_obj(), 'float16').reshape((BUFFER_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels))
	winner = np.frombuffer(s_winner.get_obj(), 'int8').reshape((N_BATCH_SETS_TOTAL, N_TURNS, 2, gv.BATCH_SZ))
	tree_probs = np.frombuffer(s_tree_probs.get_obj(), 'float32').reshape((BUFFER_SZ, gv.map_szt))
	
	######## local variables
	# BUFFER_SZ = N_BATCH_SETS * N_TURNS * 2 * gv.BATCH_SZ
	L_BUFFER_SZ = N_TURNS * 2 * gv.BATCH_SZ
	board_local = np.zeros((L_BUFFER_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels), dtype='float16')
	winner_local = np.zeros((N_TURNS, 2, gv.BATCH_SZ), dtype='int8')
	tree_probs_local = np.zeros((L_BUFFER_SZ, gv.map_szt), dtype='float32')

	if EPS_ORIG != EPS:
		#save_nm += 'EPS_%2.4f.npy' % EPS 
		save_d['EPS'] = EPS
		print 'saving to', save_nm

	### sound
	if WORKER_ID == MASTER_WORKER:
		pygame.init()
		pygame.mixer.music.load('/home/tapa/gtr-nylon22.mp3')

	######
	while True:
		#### generate training batches with `main` model
		arch.sess.run(arch.init_state)
		pu.init_tree()
		turn_start_t = time.time()
		buffer_loc_local = 0
		for turn in range(N_TURNS):
			### make move
			for player in [0,1]:
				set_weights()
				run_sim(turn, player) # using `main` model

				inds = buffer_loc_local + np.arange(gv.BATCH_SZ) # inds to save training vars at
				board_local[inds], valid_mv_map, pol = arch.sess.run([arch.imgs, arch.valid_mv_map, arch.pol['main']], feed_dict = ret_d(player)) # generate batch and valid moves

				#########
				pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
				visit_count_map = pu.choose_moves(player, np.array(pol, dtype='single'), CPUCT)[-1] # get number of times each node was visited
				
				tree_probs_local[inds] = visit_count_map / visit_count_map.sum(1)[:,np.newaxis] 

				to_coords = arch.sess.run([arch.tree_prob_visit_coord, arch.tree_prob_move_unit], feed_dict={arch.moving_player: player, 
					arch.visit_count_map: visit_count_map})[0] # make move in proportion to visit counts

				pu.register_mv(player, np.array(to_coords, dtype='int32')) # register move in tree

				###############
				
				buffer_loc_local += gv.BATCH_SZ

			pu.prune_tree(0)
			
			if (turn+1) % 2 == 0:
				print 'finished turn %i (%i sec) GPU %i batch_sets_created %i (total %i)' % (turn, time.time() - turn_start_t, WORKER_ID, batch_sets_created.value, batch_sets_created_total.value)
		
		##### create prob maps
		for player in [0,1]:
			winner_local[:, player] = arch.sess.run(arch.winner, feed_dict={arch.moving_player: player})
		
		#### set shared buffers with training variables we just generated from self-play
		with buffer_lock:
			board[buffer_loc.value:buffer_loc.value + buffer_loc_local] = board_local
			tree_probs[buffer_loc.value:buffer_loc.value + buffer_loc_local] = tree_probs_local
			winner[batch_set.value] = winner_local
			
			buffer_loc.value += buffer_loc_local
			batch_sets_created.value += 1
			batch_sets_created_total.value += 1
			batch_set.value += 1
			
			# save checkpoint
			if buffer_loc.value >= BUFFER_SZ or batch_set.value >= N_BATCH_SETS_TOTAL:
				buffer_loc.value = 0
				batch_set.value = 0
			
				# save batch only
				batch_d = {}
				for key in ['tree_probs', 'winner', 'board']:
					exec('batch_d["%s"] = copy.deepcopy(np.array(s_%s.get_obj()))' % (key, key))
				batch_save_nm = sdir + save_nm + '_batches' + str(batch_sets_created_total.value)
				np.save(batch_save_nm, batch_d)
				print 'saved', batch_save_nm
				batch_d = {}


		################ train/eval/test
		if WORKER_ID == MASTER_WORKER and batch_sets_created.value >= N_BATCH_SETS_BLOCK and batch_sets_created_total.value >= N_BATCH_SETS_MIN:
			########### train
			with buffer_lock:
				if batch_sets_created_total.value < (N_BATCH_SETS_MIN + N_BATCH_SETS_BLOCK): # don't overtrain on the initial set
					batch_sets_created.value = N_BATCH_SETS_BLOCK

				if batch_sets_created.value >= N_BATCH_SETS_TOTAL: # if for some reason master worker gets delayed
					batch_sets_created.value = N_BATCH_SETS_BLOCK

				board_c = np.array(board, dtype='single')
				winner_rc = np.array(winner.ravel(), dtype='single')
				
				valid_entries = np.prod(np.isnan(tree_probs) == False, 1) * np.nansum(tree_probs, 1) # remove examples with nans or no probabilties
				inds_valid = np.nonzero(valid_entries)[0]
				print len(inds_valid), 'out of', BUFFER_SZ, 'valid training examples'

				for rep in range(N_REP_TRAIN):
					random.shuffle(inds_valid)
					for batch in range(N_TURNS * batch_sets_created.value):
						inds = inds_valid[batch*gv.BATCH_SZ + np.arange(gv.BATCH_SZ)]

						board2, tree_probs2 = pu.rotate_reflect_imgs(board_c[inds], tree_probs[inds]) # rotate and reflect board randomly

						train_dict = {arch.imgs32: board2,
								arch.pol_target: tree_probs2,
								arch.val_target: winner_rc[inds]}

						val_mean_sq_err_tmp, pol_cross_entrop_err_tmp, val_pearsonr_tmp = \
														arch.sess.run(bp_eval_nodes, feed_dict=train_dict)[1:]

						# update logs
						val_mean_sq_err += val_mean_sq_err_tmp
						pol_cross_entrop_err += pol_cross_entrop_err_tmp
						val_pearsonr += val_pearsonr_tmp
						global_batch += 1
						err_denom += 1

				batch_sets_created.value = 0
		
			############### `eval` against prior version of self (`main`)
			set_eval16_to_eval32_start_eval() # update `eval` tf and shared copies to follow backprop (`eval32`)
			eval_model() # run match(es)
			with eval_stats_lock:
				print '-------------------'
				model_outperforms, self_eval_perc = print_eval_stats()
				print '------------------'
			if model_outperforms: # update `eval` AND `main` both tf and shared copies to follow backprop
				set_all_to_eval32_and_get()

			##### network evaluation against random player and GNU Go
			global_batch_evald = global_batch
			global_batch_saved = global_batch
			t_eval = time.time()
			print 'evaluating nn'

			d = ret_d(0)
			
			################## monitor training progress:
			# test `eval` against GNU Go and a player that makes only random moves
			for nm, N_GMS_L in zip(['nn','tree'], [[N_EVAL_NN_GNU_GMS, N_EVAL_NN_GMS], [N_EVAL_TREE_GMS, N_EVAL_TREE_GNU_GMS]]):
				for gnu, N_GMS in zip([True,False], N_GMS_L):
					if N_GMS == 0:
						continue
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
									to_coords = arch.sess.run([arch.nn_prob_to_coords_valid_mvs['eval'], arch.nn_prob_move_unit_valid_mvs['eval']], feed_dict=d)[0]
								else:
									to_coords = arch.sess.run([arch.nn_max_prob_to_coords_valid_mvs['eval'], arch.nn_max_prob_move_unit_valid_mvs['eval']], feed_dict=d)[0]


							board_tmp2 = arch.sess.run(arch.gm_vars['board'])
							n_mvs += board_tmp.sum() - board_tmp2.sum()

							# move opposing player
							if gnu:
								gt.move_nn(to_coords) 

								# mv gnugo
								ai_to_coords = gt.move_ai()
								arch.sess.run(arch.imgs, feed_dict={arch.moving_player: 1})
								arch.sess.run(arch.nn_max_move_unit['eval'], feed_dict={arch.moving_player: 1, arch.nn_max_to_coords['eval']: ai_to_coords})
							else:
								arch.sess.run(arch.imgs, feed_dict = ret_d(1))
								arch.sess.run(arch.move_random_ai, feed_dict = ret_d(1))
		
							boards[key][turn] = arch.sess.run(arch.gm_vars['board'])

							if nm == 'tree':
								pu.prune_tree(0)
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

			pol, pol_pre = arch.sess.run([arch.pol['eval'], arch.pol_pre['eval']], feed_dict={arch.moving_player: 0})

			##### log
			log['val_mean_sq_err'].append ( val_mean_sq_err / err_denom )
			log['pol_cross_entrop'].append( pol_cross_entrop_err / err_denom )
			log['val_pearsonr'].append( val_pearsonr / err_denom )
			log['opt_batch'].append( global_batch )

			log['pol_max_pre'].append( np.median(pol_pre.max(1)) )
			log['pol_max'].append( np.median(pol.max(1)) )

			log['self_eval_win_rate'].append( np.single(eval_games_won.value) / (eval_batch_sets_played.value*gv.BATCH_SZ) )
			log['model_promoted'].append( model_outperforms )

			log['self_eval_perc'].append( self_eval_perc )

			val_mean_sq_err = 0
			pol_cross_entrop_err = 0
			val_pearsonr = 0
			err_denom = 0
		
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

			# play sound
			if os.path.isfile('/home/tapa/play_sound.txt'):
				pygame.mixer.music.play()
		
		############# save
		if WORKER_ID == MASTER_WORKER:
			with buffer_lock:
				# update state vars
				#shared_nms = ['buffer_loc', 'batch_sets_created', 'batch_set', 's_board', 's_winner', 's_tree_probs', 'weights_changed', 'buffer_lock', 'weights_lock', 'save_nm', 'new_model', 'weights']
				for key in state_vars + training_ex_vars:
					if key in ['buffer_loc', 'batch_sets_created', 'batch_sets_created_total', 'batch_set', 'eval_games_won', 'eval_batch_sets_played']:
						exec('save_d["%s"] = %s.value' % (key, key))
					elif key in ['tree_probs', 'winner', 'board']:
						exec('save_d["%s"] = copy.deepcopy(np.array(s_%s.get_obj()))' % (key, key))
					else:
						exec('save_d["%s"] = %s' % (key, key))
			
			save_nms = [save_nm]
			if (datetime.now() - save_t).seconds > CHKP_FREQ:
				save_nms += [save_nm + str(datetime.now())]
				save_t = datetime.now()
			
			for nm in save_nms:
				np.save(sdir + nm, save_d)
				arch.saver.save(arch.sess, sdir + nm)
			
			print sdir + nm, 'saved'


####################################################################################################################

if save_nm is None:
	new_model = True # set `eval32` to `main`, and `eval` float16 copies

	##### weightings on individual loss terms:
	LSQ_LAMBDA = 0
	LSQ_REG_LAMBDA = 0
	POL_CROSS_ENTROP_LAMBDA = 1
	VAL_LAMBDA = .025
	VALR_LAMBDA = 0
	L2_LAMBDA = 1e-3 # weight regularization 
	CPUCT = 1
	
	N_REP_TRAIN = 5 # number of times more to backprop over training examples (reflections/rotations)
	
	N_BATCH_SETS_BLOCK = 7
	N_BATCH_SETS_TOTAL = 7*5 # number of batch sets to store in training buffer
	N_BATCH_SETS_MIN = N_BATCH_SETS_TOTAL

	batch_set = 0
	batch_sets_created = 0
	batch_sets_created_total = 0
	buffer_loc = 0

	GATE_THRESH = .5
	N_GATE_BATCH_SETS = 1

	##### model parameters
	N_LAYERS = 5 #10 # number of model layers
	FILTER_SZS = [3]*N_LAYERS
	STRIDES = [1]*N_LAYERS
	F = 128 # number of filters
	N_FILTERS = [F]*N_LAYERS
	N_FC1 = 128 # number of units in fully connected layer
	
	
	EPS = 2e-1 # backprop step size
	MOMENTUM = .9

	N_SIM = 800 # number of simulations at each turn
	N_TURNS = 32 # number of moves per player per game

	#### training buffers
	BUFFER_SZ = N_BATCH_SETS_TOTAL * N_TURNS * 2 * gv.BATCH_SZ

	board = np.zeros((BUFFER_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels),  dtype='float16')
	winner = np.zeros((N_BATCH_SETS_TOTAL, N_TURNS, 2, gv.BATCH_SZ), dtype='int8')
	tree_probs = np.zeros((BUFFER_SZ, gv.map_szt), dtype='float32')

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

	save_nm = 'go_%1.4fEPS_%iGMSZ_%iN_SIM_%iN_TURNS_%iN_FILTERS_%iN_LAYERS_%iN_BATCH_SETS_TOTAL_%i_N_BATCH_SET_MIN_%iN_REP_TRN_trainbug.npy' % \
		(EPS, gv.n_rows, N_SIM, N_TURNS, N_FILTERS[0], N_LAYERS, N_BATCH_SETS_TOTAL, N_BATCH_SETS_MIN, N_REP_TRAIN)

	boards = {}; scores = {} # eval
	save_d = {}
	for key in save_vars:
		exec('save_d["%s"] = %s' % (key,key))
	save_d['script_nm'] = __file__

	global_batch = 0
	global_batch_saved = 0
	global_batch_evald = 0
	save_counter = 0

	run_time = datetime.now() - datetime.now()

	log = {}
	for key in logs:
		log[key] = []

	########## save
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
else:
	new_model = False # prevent `main` from being set to `eval32` at loading

	save_d = np.load(sdir + save_nm, allow_pickle=True).item()

	for key in save_vars + state_vars + training_ex_vars:
		if key == 'save_nm':
			continue
		exec('%s = save_d["%s"]' % (key,key))

print save_nm

################### shared memory variables

###### self play from `eval` model used for training `eval32`:
s_board = mp.Array('h', board.ravel()) # shape: (BUFFER_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels)
s_winner = mp.Array('b', winner.ravel()) # (N_BATCH_SETS_TOTAL, N_TURNS, 2, gv.BATCH_SZ)
s_tree_probs = mp.Array('f', tree_probs.ravel()) # (BUFFER_SZ, gv.map_szt)

# indices, counters, & flags 
buffer_loc = mp.Value('i', buffer_loc) # index into above ^ training vars
weights_changed = mp.Value('i', 0) # 0 = no change, 1 = changed
batch_sets_created = mp.Value('i', batch_sets_created)
batch_sets_created_total = mp.Value('i', batch_sets_created_total)
batch_set = mp.Value('i', batch_set)

# evaluation (`eval` vs `main` benchmark testing to see when to update `main` to the current `eval32` backprop weights)
scope_next = mp.Value('i', 0) # alternates between 0,1 during model evaluation to dictate if `eval` or `main` starts 1st
eval_games_won = mp.Value('i', 0)
eval_batch_sets_played = mp.Value('i', 2*N_GATE_BATCH_SETS)
eval_batch_sets_main_first = mp.Value('i', 0)

buffer_lock = mp.Lock()
weights_lock = mp.Lock()
eval_stats_lock = mp.Lock()

weights = []; weights_eval = []

###### launch pool
cmd = 'p = mp.Pool(initializer=init, initargs=('
for nm in shared_nms:
	cmd += nm + ', '
cmd += '))'

### get weight shapes
exec(cmd)
weight_shapes = p.map(worker_save_shapes, [0])[0]
p.close()

for s in weight_shapes:
	weights.append( mp.Array('h', np.zeros(np.prod(s), dtype='float16')) )
	weights_eval.append( mp.Array('h', np.zeros(np.prod(s), dtype='float16')) )

######## run
exec(cmd)
p.map(worker, GPU_LIST)

#### dbg
'''cmd = 'init('
for nm in shared_nms:
	cmd += nm + ', '
cmd += ')'
exec(cmd)
worker(0)
'''
