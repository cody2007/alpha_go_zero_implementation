import copy
import os
import pygame
import numpy as np
from numpy import sqrt
from pygame.locals import *
import time
import global_vars as gv
import tensorflow as tf
import architectures.tree_tf_op_multi as arch
import gnu_go_test as gt

########################################################## configuration:
save_nm = 'models/go_0.2000EPS_7GMSZ_800N_SIM_32N_TURNS_128N_FILTERS_5N_LAYERS_35N_BATCH_SETS_TOTAL_35_N_BATCH_SETS_MIN_5N_REP_TRAIN.npy'

# load the following variables from the model .npy file:
save_vars = ['LSQ_LAMBDA', 'LSQ_REG_LAMBDA', 'POL_CROSS_ENTROP_LAMBDA', 'VAL_LAMBDA', 'VALR_LAMBDA', 'L2_LAMBDA',
	'FILTER_SZS', 'STRIDES', 'N_FILTERS', 'N_FC1', 'EPS', 'MOMENTUM', 'SAVE_FREQ', 'N_SIM', 
	'N_TURNS', 'CPUCT']

save_d = np.load(save_nm, allow_pickle=True).item()
for key in save_vars:
	if key == 'save_nm':
		continue
	exec('%s = save_d["%s"]' % (key,key))

########## over-write number of simulations previously used:
N_SIM = 2000 #500

net = 'eval32'
#net = 'eval'
#net = 'main'

run_one_pass_only = True # run only the network (no tree search)
#run_one_pass_only = False # make moves from the tree search

if run_one_pass_only == False:
	import py_util.py_util as pu

TURN_MIN = 5 # if we are near the max turns the network was trained on (N_TURNS), how much farther do we simulate?
NET_PLAYER = 0 # 0: the network plays first, 1: GNU Go plays first

############## load model, init variables
DEVICE = '/gpu:0'
arch.init_model(DEVICE, N_FILTERS, FILTER_SZS, STRIDES, N_FC1, EPS, MOMENTUM,
		LSQ_LAMBDA, LSQ_REG_LAMBDA, POL_CROSS_ENTROP_LAMBDA, VAL_LAMBDA, VALR_LAMBDA, L2_LAMBDA, training=False)

arch.saver.restore(arch.sess, save_nm)
arch.sess.run(arch.init_state)

visit_count_map = np.zeros((gv.n_rows, gv.n_cols), dtype='int32')

def ret_d(player): # return dictionary for input into tensor flow
	return {arch.moving_player: player}

def run_sim(turn, starting_player): # simulate game forward
	t_start = time.time()
	arch.sess.run(arch.session_backup)
	pu.session_backup()

	for sim in range(N_SIM):
		# backup then make next move
		# (this loop, iterates over one full game-play from present turn)
		for turn_sim in range(turn, np.max((N_TURNS+1, turn+TURN_MIN))):
			for player in [0,1]:
				if turn_sim == turn and starting_player == 1 and player == 0: # skip player 0, has already moved
					continue

				# get valid moves, network policy and value estimates:
				valid_mv_map, pol, val = arch.sess.run([arch.valid_mv_map, arch.pol[net], arch.val[net]], feed_dict=ret_d(player))

				# backup visit Q values
				if turn_sim != turn:
					pu.backup_visit(player, np.array(val, dtype='single'))

				pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
				to_coords = pu.choose_moves(player, np.array(pol, dtype='single'), CPUCT)[0] # choose moves based on policy and Q values (latter of which already stored in tree)
				pu.register_mv(player, np.array(to_coords, dtype='int32')) # register move in tree

				arch.sess.run(arch.move_frm_inputs, feed_dict={arch.moving_player: player, arch.to_coords_input: to_coords}) # move network (update GPU vars)
		
		# backup terminal state
		winner = np.array(arch.sess.run(arch.winner, feed_dict=ret_d(0)), dtype='single')
		pu.backup_visit(0, winner)
		pu.backup_visit(1, -winner)

		# return move back to previous node in tree
		arch.sess.run(arch.session_restore)
		pu.session_restore()
	
		# print progress
		if sim % 20 == 0:
			print 'simulation: ', sim, ' (%i sec)' % (time.time() - t_start)



#################################
t_start = time.time()
board = np.zeros((N_TURNS, 2, gv.BATCH_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels), dtype='float16')
winner = np.zeros((N_TURNS, gv.BATCH_SZ), dtype='int')
scores = np.zeros((N_TURNS, gv.BATCH_SZ), dtype='int')

arch.sess.run(arch.init_state)
if run_one_pass_only == False:
	pu.init_tree()

gt.init_board(arch.sess.run(arch.gm_vars['board']))
gt.move_nn(np.ones(gv.BATCH_SZ, dtype='int')*-1) # when NET_PLAYER=1, for some reason GnuGo doesn't respond unless we pass the first move

turn_start_t = time.time()
for turn in range(N_TURNS):
	for player in [0,1]:
		# network's turn
		if player == NET_PLAYER:
			
			#### make most probable mv, do not use tree search
			if run_one_pass_only:
				# 'eval32' movement ops were not defined, so get policy, from network, and then use the ops in 'eval' (where it was defined)
				d = ret_d(player)
				imgs = arch.sess.run(arch.imgs, feed_dict=d)
				d[arch.imgs32] = np.asarray(imgs, dtype='float')
				pol = arch.sess.run(arch.pol[net], feed_dict=d)	
				d[arch.pol['eval']] = pol
				
				board[turn, player] = imgs
				
				if turn == 0: # choose in proportion to probability
					to_coords = arch.sess.run([arch.nn_prob_to_coords_valid_mvs['eval'], arch.nn_prob_move_unit_valid_mvs['eval']], feed_dict=d)[0]
				else:
					to_coords = arch.sess.run([arch.nn_max_prob_to_coords_valid_mvs['eval'], arch.nn_max_prob_move_unit_valid_mvs['eval']], feed_dict=d)[0]

			##### use tree search
			else:
				run_sim(turn, player)

				board[turn, player], valid_mv_map, pol = arch.sess.run([arch.imgs, arch.valid_mv_map, arch.pol[net]], feed_dict = ret_d(player)) # generate batch and valid moves

				#########
				pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
				visit_count_map = pu.choose_moves(player, np.array(pol, dtype='single'), CPUCT)[-1] # get number of times each node was visited

				if turn == 0:
					to_coords = arch.sess.run([arch.tree_prob_visit_coord, arch.tree_prob_move_unit], feed_dict={arch.moving_player: player, 
						arch.visit_count_map: visit_count_map})[0] # make move in proportion to visit counts
				else:
					to_coords = arch.sess.run([arch.nn_max_prob_to_coords_valid_mvs[net], arch.nn_max_prob_move_unit_valid_mvs[net]], feed_dict={arch.moving_player: player,
							arch.pol[net]: visit_count_map})[0]
						
			gt.move_nn(to_coords) # tell gnugo where the network moved
		
		# gnugo's turn
		else:
			# mv gnugo
			board[turn, player], valid_mv_map = arch.sess.run([arch.imgs, arch.valid_mv_map], feed_dict = ret_d(player)) # generate batch and valid moves
			
			# register valid moves in tree:
			if run_one_pass_only == False:
				pu.add_valid_mvs(player, valid_mv_map)

			to_coords = gt.move_ai() # get move from gnu go
			
			# update gpu game state w/ move:
			arch.sess.run(arch.nn_max_move_unit['eval'], feed_dict={arch.moving_player: player, arch.nn_max_to_coords['eval']: to_coords})
		
		print turn, player
		
		# register move in tree:
		if run_one_pass_only == False:
			pu.register_mv(player, np.array(to_coords, dtype='int32'))
	
	winner[turn], scores[turn] = arch.sess.run([arch.winner, arch.score], feed_dict={arch.moving_player: NET_PLAYER})

	# prune tree
	if run_one_pass_only == False and turn != (N_TURNS-1):
		pu.prune_tree(0) # 0: prune all games in batch, 1: prune only first game
	
	if (turn+1) % 2 == 0:
		print 'eval finished turn %i (%i sec)' % (turn, time.time() - turn_start_t)


####### printing
res, score = arch.sess.run([arch.winner, arch.score], feed_dict={arch.moving_player: NET_PLAYER})
if run_one_pass_only:
	match_str = 'network run-once (per turn) mode'
else:
	match_str = 'using self-play w/ {} playout batches / turn', N_SIM

print 'wins', (res == 1).sum(), (res == 1).sum() / 128., 'ties', (res == 0).sum(), 'opp wins', (res == -1).sum(), match_str


######### save results to npy file
fname = '/tmp/'
if run_one_pass_only:
	fname += 'test_one_pass_vs_gnu.npy'
else:
	fname += 'test_%i_N_SIM_vs_gnu.npy' % N_SIM
	print N_SIM

np.save(fname, {'run_one_pass_only': run_one_pass_only, 'N_SIM': N_SIM, 'board': board,
		'res': res, 'score': score, 'winner': winner, 'scores': scores})

