import copy
import os
import pygame
import numpy as np
from numpy import sqrt
from pygame.locals import *
import time
import global_vars as gv
import tensorflow as tf
import architectures.tree_tf_op as arch
import py_util.py_util as pu

########################################################## configuration:
#save_nm = 'models/go_cpu_tree_0.200000EPS_7GMSZ_1000N_SIM_0.001000L2_LAMBDA_0.900000MOMENTUM_0.025000VAL_LAMBDA_1.000000CPUCT_20N_TURNS_128N_FILTERS_EPS0.110000_EPS0.020000.npy'
#save_nm = 'models/go_cpu_tree_0.200000EPS_7GMSZ_1000N_SIM_0.001000L2_LAMBDA_0.900000MOMENTUM_0.025000VAL_LAMBDA_1.000000CPUCT_20N_TURNS_128N_FILTERS_EPS0.110000_EPS0.020000_EPS0.010000.npy'
save_nm = 'models/go_0.2000EPS_7GMSZ_1000N_SIM_35N_TURNS_128N_FILTERS_5N_LAYERS_5N_BATCH_SETS.npy'
# ^ set save_nm = None if you want to start training a new model

#run_net = True # run only the network (no tree search)
run_net = False # make moves from the tree search

show_txt = False # don't show statistics of each move (Q and P values, visit counts)
#show_txt = True

# load the following variables from the model .npy file:
save_vars = ['LSQ_LAMBDA', 'LSQ_REG_LAMBDA', 'POL_CROSS_ENTROP_LAMBDA', 'VAL_LAMBDA', 'VALR_LAMBDA', 'L2_LAMBDA',
	'FILTER_SZS', 'STRIDES', 'N_FILTERS', 'N_FC1', 'EPS', 'MOMENTUM', 'SAVE_FREQ', 'N_SIM', 'N_TURNS', 'CPUCT', 'DIR_A']
save_d = np.load(save_nm).item()
for key in save_vars:
	if key == 'save_nm':
		continue
	exec('%s = save_d["%s"]' % (key,key))

########## over-write number of simulations previously used:
N_SIM = 100#0

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

Q_map = np.zeros((gv.n_rows, gv.n_cols), dtype='single')
Q_map_next = np.zeros_like(Q_map) # Q values for the move after the current (assuming you make the move the network predicts you will)
P_map = np.zeros_like(Q_map)
P_map_next = np.zeros_like(Q_map)
visit_count_map = np.zeros((gv.n_rows, gv.n_cols), dtype='int32')
visit_count_map_next = np.zeros_like(visit_count_map)

def ret_d(player): # return dictionary for input into tensor flow
	return {arch.moving_player: player, arch.dir_a: DIR_A, arch.dir_pre: dir_pre}

def ret_stats(player): # return Q map, P map, and visit count maps
	pol = np.zeros((gv.BATCH_SZ, gv.map_szt), dtype='single')
	pol[:,0] = 1
	Q_map, P_map, visit_count_map = pu.choose_moves(player, pol, CPUCT)[1:]

	Q_map = Q_map.reshape((gv.BATCH_SZ, gv.n_rows, gv.n_cols))[0]
	P_map = P_map.reshape((gv.BATCH_SZ, gv.n_rows, gv.n_cols))[0]
	visit_count_map = visit_count_map.reshape((gv.BATCH_SZ, gv.n_rows, gv.n_cols))[0]

	return Q_map, P_map, visit_count_map


# move neural network
def nn_mv():
	global Q_map, P_map, visit_count_map, valid_mv_map, pol
	global Q_map_next, P_map_next, visit_count_map_next, to_coords
	
	t_start = time.time()
	arch.sess.run(arch.session_backup)
	#pu.init_tree()
	pu.session_backup()

	if run_net:
		if turn == 0:	
			arch.sess.run(arch.nn_prob_move_unit_valid_mvs, feed_dict=ret_d(0))
		else:
			arch.sess.run(arch.nn_max_prob_move_unit_valid_mvs, feed_dict=ret_d(0))

		Q_map, P_map, visit_count_map = ret_stats(0)
	else:
		for sim in range(N_SIM):
			'''# initial moves
			for player in [0,1]:
				valid_mv_map, pol = arch.sess.run([arch.valid_mv_map, arch.pol], feed_dict=ret_d(player))

				pu.add_valid_mvs(player, valid_mv_map)
				to_coords = pu.choose_moves(player, pol, CPUCT)[0]
				pu.register_mv(player, to_coords)

				arch.sess.run(arch.move_frm_inputs, feed_dict={arch.moving_player: player, arch.to_coords_input: to_coords})'''
			
			# backup then make next move
			for turn_sim in range(turn, N_TURNS+1):
				for player in [0,1]:
					valid_mv_map, pol, val = arch.sess.run([arch.valid_mv_map, arch.pol, arch.val], feed_dict=ret_d(player))

					if turn_sim != turn:
						pu.backup_visit(player, val)

					pu.add_valid_mvs(player, valid_mv_map)
					to_coords = pu.choose_moves(player, pol, CPUCT)[0]
					pu.register_mv(player, to_coords)

					arch.sess.run(arch.move_frm_inputs, feed_dict={arch.moving_player: player, arch.to_coords_input: to_coords})
			
			# backup terminal state
			for player in [0,1]:
				winner = arch.sess.run(arch.winner, feed_dict=ret_d(player))
				pu.backup_visit(player, winner)

			arch.sess.run(arch.session_restore)
			pu.session_restore()

			if sim % 20 == 0:
				'''Q_map, P_map, visit_count_map = ret_stats(0)
				arch.sess.run(arch.tree_det_move_unit, feed_dict = ret_d(0))
				Q_map_next, P_map_next, visit_count_map_next = ret_stats(1)

				arch.sess.run(arch.session_restore)
				pu.session_restore()

				draw(True)
				pygame.display.set_caption('%i %2.1f' % (sim, time.time() - t_start))
				'''
				print 'simulation', sim, 'total elapsed time', time.time() - t_start

		### make move
		Q_map, P_map, visit_count_map = ret_stats(0) 

		valid_mv_map, pol = arch.sess.run([arch.imgs, arch.valid_mv_map, arch.pol], feed_dict = ret_d(0))[1:]
			
		#########
		pu.add_valid_mvs(player, valid_mv_map)
		visit_count_map = pu.choose_moves(player, pol, CPUCT)[-1]
			
		to_coords = arch.sess.run([arch.tree_det_visit_coord, arch.tree_det_move_unit], feed_dict={arch.moving_player: 0, 
			arch.visit_count_map: visit_count_map, arch.dir_pre: dir_pre, arch.dir_a: DIR_A})[0]

		pu.register_mv(player, to_coords)

		pu.prune_tree()
		print time.time() - t_start

	return arch.sess.run(arch.gm_vars['board'])[0]


##################### display
psz = 50 # size to display pieces
pszh = psz/2.
n_txt_rows = 4
window_sz = (psz*gv.n_rows, psz*gv.n_cols)

BLACK = (0,)*3
LINE_WIDTH = 2
turn = 0

windowSurface = pygame.display.set_mode(window_sz, 0, 32)
pygame.display.set_caption('Go GUI')

pygame.init()
basicFont = pygame.font.SysFont(None, 22)

whitep = pygame.image.load('notebooks/go_white.png')
blackp = pygame.image.load('notebooks/go_black.png')
blank = pygame.image.load('notebooks/go_blank.png')

whitep = pygame.transform.scale(whitep, (psz, psz))
blackp = pygame.transform.scale(blackp, (psz, psz))
blank = pygame.transform.scale(blank, window_sz)


centers = np.arange(gv.n_rows)*psz + pszh
to_coords_manual = -np.ones(gv.BATCH_SZ, dtype='int32')

board = np.zeros((gv.n_rows, gv.n_cols), dtype='int8')

# draw board and optionally text
def draw(update=False):
	windowSurface.blit(blank, (0,0))

	# draw lines
	for i in range(gv.n_rows):
		pygame.draw.line(windowSurface, BLACK, (0, i*psz + pszh), (window_sz[0], i*psz + pszh), LINE_WIDTH)
		pygame.draw.line(windowSurface, BLACK, (i*psz + pszh, 0), (i*psz + pszh, window_sz[1]), LINE_WIDTH)

	for i in range(gv.n_rows):
		for j in range(gv.n_cols):
			coord = np.asarray((i*psz, j*psz))
			if board[i,j] == 1:
				windowSurface.blit(blackp, coord)
			elif board[i,j] == -1:
				windowSurface.blit(whitep, coord)

			if P_map[i,j] != 0 and show_txt:
				visit_total = visit_count_map.sum()
				rc = np.int(np.min((255, 3*255.*visit_count_map[i,j] / np.single(visit_total))))
				bgc = [rc, 0, 0]
				fc = [255,255,255]

				txt = '%1.2f + %1.2f' % (Q_map[i,j], P_map[i,j])
				tsz = np.asarray(basicFont.size(txt), dtype='single')
				tcoord = coord + pszh - np.asarray([tsz[0]/2., n_txt_rows*tsz[1]/2])
				pygame.draw.rect(windowSurface, bgc, [tcoord, tsz])

				text = basicFont.render(txt, True, fc)
				windowSurface.blit(text, tcoord)
				tsz1 = copy.deepcopy(tsz)

				txt = '%1.2f  %i' % (Q_map[i,j]+P_map[i,j], visit_count_map[i,j])
				tsz = np.asarray(basicFont.size(txt), dtype='single')
				tcoord = coord + pszh - np.asarray([tsz[0]/2., n_txt_rows*tsz[1]/2])
				tcoord[1] += tsz1[1]
				pygame.draw.rect(windowSurface, bgc, [tcoord, tsz])

				text = basicFont.render(txt, True, fc)
				windowSurface.blit(text, tcoord)
				tsz2 = copy.deepcopy(tsz)
			else:
				tsz1 = tsz2 = [0,0]
			
			if P_map_next[i,j] and show_txt:
				visit_total = visit_count_map_next.sum()
				rc = np.int(np.min((255, 3*255.*visit_count_map_next[i,j] / np.single(visit_total))))
				bgc = [0, rc, 0]
				fc = [255,255,255]
				
				txt = '%1.2f + %1.2f' % (Q_map_next[i,j], P_map_next[i,j])
				tsz = np.asarray(basicFont.size(txt), dtype='single')
				tcoord = coord + pszh - np.asarray([tsz[0]/2., n_txt_rows*tsz[1]/2])
				tcoord[1] += tsz1[1] + tsz2[1]
				pygame.draw.rect(windowSurface, bgc, [tcoord, tsz])

				text = basicFont.render(txt, True, fc)
				windowSurface.blit(text, tcoord)
				tsz3 = copy.deepcopy(tsz)

				txt = '%1.2f  %i' % (Q_map_next[i,j]+P_map_next[i,j], visit_count_map_next[i,j])
				tsz = np.asarray(basicFont.size(txt), dtype='single')
				tcoord = coord + pszh - np.asarray([tsz[0]/2., n_txt_rows*tsz[1]/2])
				tcoord[1] += tsz1[1] + tsz2[1] + tsz3[1]
				pygame.draw.rect(windowSurface, bgc, [tcoord, tsz])

				text = basicFont.render(txt, True, fc)
				windowSurface.blit(text, tcoord)


	if update:
		pygame.display.update()

draw(update=True)
board = nn_mv()
draw(update=True)

while True:
	event = pygame.event.wait()

	if event.type == MOUSEBUTTONUP:
		mouse_pos = np.asarray(event.pos)
		x = np.argmin((mouse_pos[0] - centers)**2)
		y = np.argmin((mouse_pos[1] - centers)**2)
	
		to_coords_manual[0] = x*gv.n_cols + y
	
		board_prev = arch.sess.run(arch.gm_vars['board'])[0]

		imgs, valid_mv_map = arch.sess.run([arch.imgs, arch.valid_mv_map], feed_dict={arch.moving_player: 1})
		
		arch.sess.run(arch.nn_max_move_unit, feed_dict={arch.moving_player: 1, arch.nn_max_to_coords: to_coords_manual})

		board = arch.sess.run(arch.gm_vars['board'])[0]
		if board_prev.sum() == board.sum(): # invalid move
			print 'invalid mv'
			continue
		
		pu.add_valid_mvs(1, valid_mv_map) # register valid moves in tree
		pu.register_mv(1, to_coords_manual)

		win_tmp, score_tmp = arch.sess.run([arch.winner, arch.score], feed_dict={arch.moving_player: 1})
		print 'turn %i, winner %i, score %i' % (turn, win_tmp[0], score_tmp[0])

		draw(update=True)
		board = nn_mv()
		draw(update=True)
		turn += 1
		
		win_tmp, score_tmp = arch.sess.run([arch.winner, arch.score], feed_dict={arch.moving_player: 1})
		print 'turn %i, winner %i, score %i' % (turn, win_tmp[0], score_tmp[0])

	if event.type == QUIT:
		pygame.display.quit()
		break

