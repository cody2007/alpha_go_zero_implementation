import copy
import os
import pygame
import numpy as np
from numpy import sqrt
from pygame.locals import *
import time
from datetime import datetime
import global_vars as gv
import tensorflow as tf
import architectures.tree_tf_op_multi as arch

########################################################## configuration:
save_nm = 'models/go_0.2000EPS_7GMSZ_800N_SIM_32N_TURNS_128N_FILTERS_5N_LAYERS_35N_BATCH_SETS_TOTAL_35_N_BATCH_SETS_MIN_5N_REP_TRAIN.npy'

net = 'eval32'
#net = 'eval'
#net = 'main'

run_one_pass_only = True # run only the network (no tree search)
#run_one_pass_only = False # make moves from the tree search

show_txt = False # don't show statistics of each move (Q and P values, visit counts) -- toggle w/ right click after network makes move

# load the following variables from the model .npy file:
save_vars = ['LSQ_LAMBDA', 'LSQ_REG_LAMBDA', 'POL_CROSS_ENTROP_LAMBDA', 'VAL_LAMBDA', 'VALR_LAMBDA', 'L2_LAMBDA',
	'FILTER_SZS', 'STRIDES', 'N_FILTERS', 'N_FC1', 'EPS', 'MOMENTUM', 'SAVE_FREQ', 'N_SIM', 'N_TURNS', 'CPUCT']
save_d = np.load(save_nm, allow_pickle=True).item()
for key in save_vars:
	if key == 'save_nm':
		continue
	exec('%s = save_d["%s"]' % (key,key))

if run_one_pass_only == False:
	import py_util.py_util as pu

########## over-write number of simulations previously used:
# (stop self-play when both of these (the next two) conditions is met)
SIM_MIN = 2000
TIME_MIN = 1 # time spent running self-play exceeds this (minutes)

###
TURN_MIN = 5 # if we are near the max turns the network was trained on (N_TURNS), how much farther do we simulate?
CPUCT = 1
NET_PLAYER = 0 # 0: the network plays first, 1: you play first

def human_player():
	global NET_PLAYER
	assert NET_PLAYER == 1 or NET_PLAYER == 0
	return 1 - NET_PLAYER

###############################################################################
save_screenshot_flag = True

img_sdir = 'go_games_imgs/'
img_sdir += datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.system('mkdir ' + img_sdir)
os.system("echo %s > %s/model_location.txt" % (save_nm, img_sdir))

############## load model, init variables
DEVICE = '/gpu:0'
arch.init_model(DEVICE, N_FILTERS, FILTER_SZS, STRIDES, N_FC1, EPS, MOMENTUM,
		LSQ_LAMBDA, LSQ_REG_LAMBDA, POL_CROSS_ENTROP_LAMBDA, VAL_LAMBDA, VALR_LAMBDA, L2_LAMBDA, training=False)

arch.saver.restore(arch.sess, save_nm)
arch.sess.run(arch.init_state)
if run_one_pass_only == False:
	pu.init_tree()

##### stats to print if show_txt = True
Q_map = np.zeros((gv.n_rows, gv.n_cols), dtype='single')
Q_map_next = np.zeros_like(Q_map) # Q values for the move after the current (assuming you make the move the network predicts you will)
P_map = np.zeros_like(Q_map)
P_map_next = np.zeros_like(Q_map)
visit_count_map = np.zeros((gv.n_rows, gv.n_cols), dtype='int32')
visit_count_map_next = np.zeros_like(visit_count_map)

t_init = time.time()	

def ret_d(player): # return dictionary for input into tensor flow
	return {arch.moving_player: player}

def ret_stats(player): # return Q map, P map, and visit count maps
	pol = np.zeros((gv.BATCH_SZ, gv.map_szt), dtype='float32')
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
	
	#### make most probable mv, do not use tree search
	if run_one_pass_only:
		# 'eval32' movement ops were not defined, so get policy, from network, and then use the ops in 'eval' (where it was defined)
		d = ret_d(NET_PLAYER)
		imgs = arch.sess.run(arch.imgs, feed_dict=d)
		d[arch.imgs32] = np.asarray(imgs, dtype='float')
		pol = arch.sess.run(arch.pol[net], feed_dict=d)	
		d = ret_d(NET_PLAYER)
		d[arch.pol['eval']] = pol

		if turn == 0:	
			arch.sess.run(arch.nn_prob_move_unit_valid_mvs['eval'], feed_dict=d)
		else:
			arch.sess.run(arch.nn_max_prob_move_unit_valid_mvs['eval'], feed_dict=d)

		#Q_map, P_map, visit_count_map = ret_stats(0)
	
	##### use tree search
	else:
		#pu.init_tree()
		pu.session_backup()

		sim = 0
		# each loop is one simulation
		while True:
			if ((time.time() - t_start) > TIME_MIN) and (sim >= SIM_MIN):
				break
			
			# backup then make next move
			# (this loop, iterates over one full game-play from present turn)
			for turn_sim in range(turn, np.max((N_TURNS+1, turn+TURN_MIN))):
				for player in [0,1]:
					if turn_sim == turn and human_player() == 0 and player == 0: # skip player 0 (human), has already moved
						continue

					# get valid moves, network policy and value estimates:
					valid_mv_map, pol, val = arch.sess.run([arch.valid_mv_map, arch.pol[net], arch.val[net]], feed_dict=ret_d(player))

					# backup visit Q values
					if turn_sim != turn:
						pu.backup_visit(player, np.array(val, dtype='single'))

					pu.add_valid_mvs(player, valid_mv_map) # register valid moves in tree
					to_coords = pu.choose_moves(player, np.array(pol, dtype='float32'), CPUCT)[0] # choose moves based on policy and Q values (latter of which already stored in tree)
					
					pu.register_mv(player, np.array(to_coords, dtype='int32')) # register move in tree
					arch.sess.run(arch.move_frm_inputs, feed_dict={arch.moving_player: player, arch.to_coords_input: to_coords}) # move network (update GPU vars)

			# backup terminal state
			winner = np.array(arch.sess.run(arch.winner, feed_dict=ret_d(0)), dtype='single')
			pu.backup_visit(0, winner)
			pu.backup_visit(1, -winner)
			
			# return move to previous node in tree
			arch.sess.run(arch.session_restore) # reset gpu game state
			pu.session_restore() # reset cpu tree state
			
			######################
			# print stats from tree
			if sim % 20 == 0:
				# get valid moves, network policy and value estimates:
				valid_mv_map = arch.sess.run([arch.imgs, arch.valid_mv_map], feed_dict=ret_d(NET_PLAYER))[1]
				pu.add_valid_mvs(NET_PLAYER, valid_mv_map) # register valid moves in tree
				
				visit_count_map_128 = pu.choose_moves(NET_PLAYER, np.array(pol, dtype='float32'), CPUCT)[-1] # to feed back into tf (entries for all 128 games, not just 1)
				Q_map, P_map, visit_count_map = ret_stats(NET_PLAYER) # stats we will show on screen
				
				# move network where it is estimates is its best move
				to_coords = arch.sess.run([arch.nn_max_prob_to_coords_valid_mvs[net], arch.nn_max_prob_move_unit_valid_mvs[net]], feed_dict={arch.moving_player: NET_PLAYER,
						arch.pol[net]: visit_count_map_128})[0]

				pu.register_mv(NET_PLAYER, np.asarray(to_coords, dtype='int32')) # register move in tree
				arch.sess.run(arch.move_frm_inputs, feed_dict={arch.moving_player: NET_PLAYER, arch.to_coords_input: to_coords}) # move network (update GPU vars)

				# get network tree estimates as to where it thinks you will move after it moves
				valid_mv_map = arch.sess.run([arch.imgs, arch.valid_mv_map], feed_dict=ret_d(human_player()))[1]
				pu.add_valid_mvs(human_player(), valid_mv_map) # register valid moves in tree

				Q_map_next, P_map_next, visit_count_map_next = ret_stats(human_player())

				arch.sess.run(arch.session_restore) # restore prior tf game state
				pu.session_restore() # restore prior tree

				draw(True)
				pygame.display.set_caption('%i %2.1f' % (sim, time.time() - t_start))
				
				print 'simulation: ', sim, ' (%i sec)' % (time.time() - t_start)
			
			sim += 1

		### make move
		
		# first get valid moves and current policy at board position
		valid_mv_map, pol = arch.sess.run([arch.imgs, arch.valid_mv_map, arch.pol[net]], feed_dict = ret_d(NET_PLAYER))[1:]
		pu.add_valid_mvs(NET_PLAYER, valid_mv_map) # set in tree
		
		visit_count_map_128 = pu.choose_moves(NET_PLAYER, np.array(pol, dtype='float32'), CPUCT)[-1] # to feed back into tf (entries for all 128 games, not just 1)
		Q_map, P_map, visit_count_map = ret_stats(NET_PLAYER)

		# makes moves as if this were still part of the self-play (max visit count)
		#to_coords = arch.sess.run([arch.tree_det_visit_coord, arch.tree_det_move_unit], feed_dict={arch.moving_player: 0, 
		#				arch.visit_count_map: visit_count_map})[0]
		
		# move to max visited node:
		#if turn != 0:
		to_coords = arch.sess.run([arch.nn_max_prob_to_coords_valid_mvs[net], arch.nn_max_prob_move_unit_valid_mvs[net]], feed_dict={arch.moving_player: NET_PLAYER,
						arch.pol[net]: visit_count_map_128})[0]
		
		# randomly move proportionatly to vist counts
		#else:
		#	to_coords = arch.sess.run([arch.tree_prob_visit_coord, arch.tree_prob_move_unit], feed_dict={arch.moving_player: 0, 
		#			arch.visit_count_map: visit_count_map})[0] # make move in proportion to visit counts

		pu.register_mv(NET_PLAYER, np.array(to_coords, dtype='int32'))
		
		print 'pruning...'
		pu.prune_tree(1) # 0: prune all games in batch, 1: prune only first game
		print time.time() - t_start

	print 'finished'
	return arch.sess.run(arch.gm_vars['board'])[0]

def save_screenshot(player):
	if save_screenshot_flag == False:
		return
	
	fname = "%s/%i_%i_%s_net_%s_one_pass_%i_ai_player_%i" % (img_sdir, t_init, turn, player, net, run_one_pass_only, NET_PLAYER)
	if run_one_pass_only == False:
		fname += '_%isims' % SIM_MIN
	
	pygame.image.save(windowSurface, fname + '.png')

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
basicFont = pygame.font.SysFont(None, 15) # < font size

whitep = pygame.image.load('notebooks/go_white.png')
blackp = pygame.image.load('notebooks/go_black.png')
blank = pygame.image.load('notebooks/go_blank.png')

whitep = pygame.transform.scale(whitep, (psz, psz))
blackp = pygame.transform.scale(blackp, (psz, psz))
blank = pygame.transform.scale(blank, window_sz)


centers = np.arange(gv.n_rows)*psz + pszh
to_coords_manual = -np.ones(gv.BATCH_SZ, dtype='int32')

board = np.zeros((gv.n_rows, gv.n_cols), dtype='int8')

# draw text over partially transparent background
# tcoord is the coordinate, tsz is the size, bgc is the color
def draw_txt(txt, tcoord, tsz, bgc):
	txtBgSurface = pygame.Surface(tsz)
	txtBgSurface.set_alpha(128)
	txtBgSurface.fill(bgc)
	windowSurface.blit(txtBgSurface, tcoord)
	
	fc = [255,255,255]
	text = basicFont.render(txt, True, fc)
	windowSurface.blit(text, tcoord)



# draw board and optionally text
def draw(update=False):
	windowSurface.blit(blank, (0,0))

	# draw lines
	for i in range(gv.n_rows):
		pygame.draw.line(windowSurface, BLACK, (0, i*psz + pszh), (window_sz[0], i*psz + pszh), LINE_WIDTH)
		pygame.draw.line(windowSurface, BLACK, (i*psz + pszh, 0), (i*psz + pszh, window_sz[1]), LINE_WIDTH)
	
	# loop over all positions on game board
	for i in range(gv.n_rows):
		for j in range(gv.n_cols):
			coord = np.asarray((i*psz, j*psz))
			# show pieces
			if board[i,j] == 1:
				windowSurface.blit(blackp, coord)
			elif board[i,j] == -1:
				windowSurface.blit(whitep, coord)
			
			##############
			# print tree statistics (for the network's own movement)
			if P_map[i,j] != 0 and show_txt:
				visit_total = visit_count_map.sum()
				rc = np.int(np.min((255, 3*255.*visit_count_map.reshape(gv.map_sz)[i,j] / np.single(visit_total))))
				bgc = [rc, 0, 0]
				
				# Show Q and P at each location on map
				txt = '%1.2f %1.2f' % (Q_map.reshape(gv.map_sz)[i,j], P_map.reshape(gv.map_sz)[i,j])
				tsz = np.asarray(basicFont.size(txt), dtype='single')
				tcoord = coord + pszh - np.asarray([tsz[0]/2., n_txt_rows*tsz[1]/2])
				draw_txt(txt, tcoord, tsz, bgc)
				tsz1 = copy.deepcopy(tsz)
				
				# Show Q + P, and visit_count_map
				txt = '%1.2f %i' % (Q_map.reshape(gv.map_sz)[i,j]+P_map.reshape(gv.map_sz)[i,j], visit_count_map.reshape(gv.map_sz)[i,j])
				tsz = np.asarray(basicFont.size(txt), dtype='single')
				tcoord = coord + pszh - np.asarray([tsz[0]/2., n_txt_rows*tsz[1]/2])
				tcoord[1] += tsz1[1]
				draw_txt(txt, tcoord, tsz, bgc)
				tsz2 = copy.deepcopy(tsz)
			else:
				tsz1 = tsz2 = [0,0]
			
			###############
			# print tree statistics (where the network estimates *you* will play)
			if P_map_next[i,j] and show_txt:
				visit_total = visit_count_map_next.sum()
				rc = np.int(np.min((255, 3*255.*visit_count_map_next.reshape(gv.map_sz)[i,j] / np.single(visit_total))))
				bgc = [0, rc, 0]
				fc = [255,255,255]
				
				# Show Q and P at each location on map 
				txt = '%1.2f %1.2f' % (Q_map_next.reshape(gv.map_sz)[i,j], P_map_next.reshape(gv.map_sz)[i,j])
				tsz = np.asarray(basicFont.size(txt), dtype='single')
				tcoord = coord + pszh - np.asarray([tsz[0]/2., n_txt_rows*tsz[1]/2])
				tcoord[1] += tsz1[1] + tsz2[1]
				draw_txt(txt, tcoord, tsz, bgc)
				tsz3 = copy.deepcopy(tsz)

				# Show Q + P, and visit_count_map
				txt = '%1.2f %i' % (Q_map_next.reshape(gv.map_sz)[i,j]+P_map_next.reshape(gv.map_sz)[i,j], visit_count_map_next.reshape(gv.map_sz)[i,j])
				tsz = np.asarray(basicFont.size(txt), dtype='single')
				tcoord = coord + pszh - np.asarray([tsz[0]/2., n_txt_rows*tsz[1]/2])
				tcoord[1] += tsz1[1] + tsz2[1] + tsz3[1]
				draw_txt(txt, tcoord, tsz, bgc)


	if update:
		pygame.display.update()

draw(update=True)

if NET_PLAYER == 0: # network makes first move
	board = nn_mv()
	draw(update=True)
	save_screenshot('b')

#pygame.mixer.music.load('/home/tapa/gtr-nylon22.mp3')

while True:
	event = pygame.event.wait()
	
	# move player, then move network
	if event.type == MOUSEBUTTONUP:
		
		# if right button pressed, toggle showing tree stats
		if event.button == 3:
			show_txt = not show_txt
			draw(update=True)
			continue
		
		# get player move from cursor
		mouse_pos = np.asarray(event.pos)
		x = np.argmin((mouse_pos[0] - centers)**2)
		y = np.argmin((mouse_pos[1] - centers)**2)
	
		to_coords_manual[0] = x*gv.n_cols + y
	
		board_prev = arch.sess.run(arch.gm_vars['board'])[0]

		imgs, valid_mv_map = arch.sess.run([arch.imgs, arch.valid_mv_map], feed_dict={arch.moving_player: human_player()})
		
		# make move for player
		arch.sess.run(arch.nn_max_move_unit['eval'], feed_dict={arch.moving_player: human_player(), arch.nn_max_to_coords['eval']: to_coords_manual})

		# valid?
		board = arch.sess.run(arch.gm_vars['board'])[0]
		if board_prev.sum() == board.sum(): # invalid move
			print 'invalid mv'
			continue
		
		# register in tree if not in one-pass-only mode
		if run_one_pass_only == False:
			pu.add_valid_mvs(human_player(), valid_mv_map) # register valid moves in tree
			pu.register_mv(human_player(), to_coords_manual)

		win_tmp, score_tmp = arch.sess.run([arch.winner, arch.score], feed_dict={arch.moving_player: human_player()})
		print 'you: turn %i, winner %i, score %i' % (turn, win_tmp[0], score_tmp[0])

		draw(update=True)
		save_screenshot('w')
		
		# network makes move
		board = nn_mv()
		draw(update=True)
		turn += 1
		save_screenshot('b')
		
		if run_one_pass_only == False:
			pygame.mixer.music.play()
	
		win_tmp, score_tmp = arch.sess.run([arch.winner, arch.score], feed_dict={arch.moving_player: NET_PLAYER})
		print 'network: turn %i, winner %i, score %i' % (turn, win_tmp[0], score_tmp[0])

	if event.type == QUIT:
		pygame.display.quit()
		break

