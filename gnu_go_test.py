import subprocess as sp
from subprocess import Popen, PIPE
from time import sleep
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK, read
import global_vars as gv
import numpy as np

PAUSE = .001
row_nm = 'ABCDEFGHIJKLMNOP'
colors = 'BW'
f = [None]*gv.BATCH_SZ

### start gnugo
for gm in range(gv.BATCH_SZ):
	f[gm] = sp.Popen(['gnugo', '--chinese-rules', '--seed', str(gm+1), '--play-out-aftermath', '--capture-all-dead', '--no-ko', '--never-resign', '--mode','gtp','--boardsize',str(gv.n_rows)], stdout=sp.PIPE, stdin=sp.PIPE)
	flags = fcntl(f[gm].stdout, F_GETFL) # get current p.stdout flags
	fcntl(f[gm].stdout, F_SETFL, flags | O_NONBLOCK)

def read_resp(gm):
	sleep(PAUSE)
	resp = ' '
	while resp[-1] != '\n':
		sleep(PAUSE)
		try:
			resp2 = f[gm].stdout.read()
			resp += resp2
		except:
			continue
	return resp[1:]

def req_ok(gm, cmd):
	f[gm].stdin.write(cmd)
	resp = read_resp(gm)
	assert resp[:2] == '= ', 'err reading resp gm %i, cmd %s resp %s' % (gm, cmd, resp)

def req_ok_or_illegal(gm, cmd):
	f[gm].stdin.write(cmd)
	resp = read_resp(gm)
	assert resp[:2] == '= ' or resp.find('? illegal move') != -1

def init_board(board):
	for gm in range(gv.BATCH_SZ):
		req_ok(gm, 'clear_board\n')
		for i in range(gv.n_rows):
			for j in range(gv.n_cols):
				if board[gm,i,j] == 0:
					continue
				#req_ok(gm, 'play %s %s%i\n' % (colors[np.int((board[gm,i,j]+1.)/2)], row_nm[j], gv.n_rows - i))
				f[gm].stdin.write('play %s %s%i\n' % (colors[np.int((board[gm,i,j]+1.)/2)], row_nm[j], gv.n_rows - i))

def move_nn(to_coords, moving_player=0):
	passes = to_coords == -1
	to_coords[passes] = 0
	i, j = np.unravel_index(to_coords, (gv.n_rows, gv.n_cols))
	for gm in range(gv.BATCH_SZ):
		#req_ok_or_illegal(gm, 'play %s %s%i\n' % (colors[moving_player], row_nm[j[gm]], gv.n_rows - i[gm]))
		if passes[gm]:
			cmd = 'play %s pass\n' % colors[moving_player]
		else:
			cmd = 'play %s %s%i\n' % (colors[moving_player], row_nm[j[gm]], gv.n_rows - i[gm])
		f[gm].stdin.write(cmd)

def move_ai(moving_player=1):
	ai_to_coords = -np.ones(gv.BATCH_SZ, dtype='int32')

	for gm in range(gv.BATCH_SZ):
		while True:
			try:
				f[gm].stdout.read()
				break
			except:
				j = 1

		f[gm].stdin.write('genmove %s\n' % colors[moving_player])
	
	for gm in range(gv.BATCH_SZ):
		ai_mv_orig = read_resp(gm)
		ai_mv = ai_mv_orig.split('\n\n')[-2]
		if ai_mv[:2] != '= ':
			print 'failed gm %i resp %s' % (gm, ai_mv)
			continue
		#assert ai_mv[:2] == '= ', 'gm %i resp %s' % (gm, ai_mv)
		if ai_mv.find('= PASS') != -1:
			#print 'pass ', gm
			continue
		if ai_mv.find('= resign') != -1:
			print 'resign ', gm
			#assert False
			continue
		if len(ai_mv) <= 3:
			#assert False, 'gm %i resp %s, orig %s' % (gm, ai_mv, ai_mv_orig)
			assert 'gm %i resp %s, orig %s' % (gm, ai_mv, ai_mv_orig)
			continue
		col = row_nm.find(ai_mv[2])
		assert col != -1, 'gm %i resp %s' % (gm, ai_mv)
		row = gv.n_rows - np.int(ai_mv[3:])

		ai_to_coords[gm] = row*gv.n_cols + col
	return ai_to_coords

def show_board(gm):
	f[gm].stdin.write('showboard\n')
	print read_resp(gm)


