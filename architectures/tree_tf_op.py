import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
import global_vars as gv
import os
sess = tf.InteractiveSession()

hdir = os.getenv('HOME')
tf_op = tf.load_op_library('cuda_op_kernel.so')

imgs_shape = [gv.BATCH_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels]
map_prod = np.prod(gv.map_sz)

##################### set / load vars
set_var_int32 = tf.placeholder(tf.int32, shape=[None])
set_var_int8 = tf.placeholder(tf.int8, shape=[None])

gm_var_nms = ['board', 'valid_mv_map_internal']

gm_var_placeholders = ['set_var_int8']*2

gm_vars = {}; set_gm_vars = {}

def return_vars():
	v = {}
	for var in gm_var_nms:
		exec('v["%s"] = sess.run(gm_vars["%s"])' % (var, var))
	return v

def set_vars(v):
	for var, placeholder in zip(gm_var_nms, gm_var_placeholders):
		exec('sess.run(set_gm_vars["%s"], feed_dict={%s: v["%s"].ravel()})' % (var, placeholder, var))

########################

def tf_pearsonr(val, val_target_nmean):
	val_nmean = val - tf.reduce_mean(val)
	val_target_nmean = val_target - tf.reduce_mean(val_target)
	
	val_std = tf.sqrt(tf.reduce_sum(val_nmean**2))
	val_target_std = tf.sqrt(tf.reduce_sum(val_target_nmean**2))

	return -tf.reduce_sum(val_nmean * val_target_nmean) / (val_std * val_target_std)


def init_model(N_FILTERS, FILTER_SZS, STRIDES, N_FC1, EPS, MOMENTUM, \
		LSQ_LAMBDA, LSQ_REG_LAMBDA, POL_CROSS_ENTROP_LAMBDA, VAL_LAMBDA, VALR_LAMBDA, L2_LAMBDA, WEIGHT_STD=1e-2):
	
	global convs, weights, outputs, output_nms, pol, pol_pre, pol_mean_sq_err, train_step
	global val, val_mean_sq_err, pol_loss, entrop, saver, update_ops 
	global val_pearsonr, pol_mean_sq_reg_err, loss, Q_map, P_map, visit_count_map
	global move_random_ai, init_state, nn_move_unit, nn_prob_move_unit
        global tree_to_coords, nn_max_to_coords, nn_prob_to_coords
	global tree_prob_move_unit, backup_visit, backup_visit_terminal, tree_det_move_unit
	global nn_prob_move_unit, nn_max_move_unit, tree_prob_visit_coord, tree_det_visit_coord
	global sess, imgs, valid_mv_map, pol_target, val_target, moving_player
	global gm_vars, set_gm_vars, oFC1, session_restore, session_backup
	global winner, dir_pre, dir_a
	global games_running, score, n_captures, pol_cross_entrop_err
	global nn_prob_to_coords_valid_mvs, nn_max_prob_to_coords_valid_mvs
	global nn_prob_move_unit_valid_mvs, nn_max_prob_move_unit_valid_mvs
	assert len(N_FILTERS) == len(FILTER_SZS) == len(STRIDES)

	#### init state
        init_state = tf_op.init_state()

	dir_pre = tf.placeholder(tf.float32, shape=())
	dir_a = tf.placeholder(tf.float32, shape=())

	moving_player = tf.placeholder(tf.int32, shape=())
	winner, score, n_captures = tf_op.return_winner(moving_player)

	games_running = tf.ones(gv.BATCH_SZ, dtype=tf.int8)

	session_restore = tf_op.session_restore()
	session_backup = tf_op.session_backup()

	##### vars
	for var, placeholder in zip(gm_var_nms, gm_var_placeholders):
		exec('gm_vars["%s"] = tf_op.%s()' % (var, var))
		exec('set_gm_vars["%s"] = tf_op.set_%s(%s)' % (var, var, placeholder))

	#### imgs
	imgs, valid_mv_map = tf_op.create_batch(moving_player)
	#print imgs.shape, imgs_shape
	assert imgs.shape == tf.placeholder(tf.float32, shape=imgs_shape).shape, 'tf op shape not matching global_vars'
	move_random_ai = tf_op.move_random_ai(moving_player)

	global move_frm_inputs, to_coords_input
	to_coords_input = tf.placeholder(tf.int32, shape=gv.BATCH_SZ)
	move_frm_inputs = tf_op.move_unit(to_coords_input, moving_player)

	####
        pol_target = tf.placeholder(tf.float32, shape=[gv.BATCH_SZ, map_prod])
        val_target = tf.placeholder(tf.float32, shape=[gv.BATCH_SZ])
        
        convs = []; weights = []; outputs = []; output_nms = []
	
	convs += [tf.nn.relu(tf.contrib.layers.batch_norm(tf.layers.conv2d(inputs=imgs, filters=N_FILTERS[0], kernel_size=[FILTER_SZS[0]]*2, 
		strides=[STRIDES[0]]*2, padding="same", activation=None, name='conv0',
		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA),
		bias_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA))))]

	weights += [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv0')[0]]
	outputs += [convs[-1]]
	output_nms += ['conv0']

	for i in range(1, len(N_FILTERS)):
		output_nms += ['conv' + str(i)]
		
		conv_out = tf.contrib.layers.batch_norm(\
			tf.layers.conv2d(inputs=convs[i-1], filters=N_FILTERS[i], kernel_size=[FILTER_SZS[i]]*2,
				strides=[STRIDES[i]]*2, padding="same", activation=None, name=output_nms[-1],
				kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA),
				bias_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA)))
		
		# residual bypass
		if (i % 2) == 0:
			conv_out += convs[i-2]

		convs += [tf.nn.relu(conv_out)]

		weights += [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, output_nms[-1])[0]]
		outputs += [convs[-1]]

	out_sz = np.int(np.prod(convs[-1].shape[1:]))
	convr = tf.reshape(convs[-1], [gv.BATCH_SZ, out_sz])

	################### pol
	# FC layer
	wFC1p = tf.Variable(tf.random_normal([out_sz, N_FC1], stddev=WEIGHT_STD), name='wFC1')
	bFC1p = tf.Variable(tf.random_normal([N_FC1], mean=WEIGHT_STD*2, stddev=WEIGHT_STD), name='bFC1')
	
	oFC1p = tf.nn.relu(tf.matmul(convr, wFC1p) + bFC1p)

	weights += [wFC1p]
	outputs += [oFC1p]
	output_nms += ['oFC1p']

	# FC layer
	wFC2p = tf.Variable(tf.random_normal([N_FC1, map_prod], stddev=WEIGHT_STD), name='wFC2')
	bFC2p = tf.Variable(tf.random_normal([map_prod], mean=WEIGHT_STD*2, stddev=WEIGHT_STD), name='bFC2')
	
	pol_pre = tf.nn.relu(tf.matmul(oFC1p, wFC2p) + bFC2p)

	weights += [wFC2p]
	outputs += [pol_pre]
	output_nms += ['pol_pre']
	
	pol = tf.nn.softmax(pol_pre)
	outputs += [pol]
	output_nms += ['pol']
	
	nn_max_to_coords = tf.argmax(pol_pre, 1, output_type=tf.int32)
	nn_prob_to_coords = tf_op.prob_to_coord(pol, dir_pre, dir_a) 
	nn_prob_to_coords_valid_mvs = tf_op.prob_to_coord_valid_mvs(pol)
	nn_max_prob_to_coords_valid_mvs = tf_op.max_prob_to_coord_valid_mvs(pol)

	# move unit
	nn_max_move_unit = tf_op.move_unit(nn_max_to_coords, moving_player)
	nn_prob_move_unit = tf_op.move_unit(nn_prob_to_coords, moving_player)
	nn_prob_move_unit_valid_mvs = tf_op.move_unit(nn_prob_to_coords_valid_mvs, moving_player)
	nn_max_prob_move_unit_valid_mvs = tf_op.move_unit(nn_max_prob_to_coords_valid_mvs, moving_player)

	# sq
	sq_err = tf.reduce_sum((pol - pol_target)**2, axis=1)
	pol_mean_sq_err = tf.reduce_mean(sq_err)

	# sq reg
	sq_err_reg = tf.reduce_sum(pol_pre**2, axis=1)
	pol_mean_sq_reg_err = tf.reduce_mean(sq_err_reg)

	# cross entrop
	pol_ln = tf.log(pol)
	pol_cross_entrop_err = -tf.reduce_mean(pol_target*pol_ln)

	################# val
	# FC layer
	wFC1v = tf.Variable(tf.random_normal([out_sz, N_FC1], stddev=WEIGHT_STD), name='val_wFC1')
	bFC1v = tf.Variable(tf.random_normal([N_FC1], mean=WEIGHT_STD*2, stddev=WEIGHT_STD), name='val_bFC1')
	
	#oFC1v = tf.nn.relu(tf.matmul(convr, wFC1v) + bFC1v)
	oFC1v = tf.matmul(convr, wFC1v) + bFC1v

	weights += [wFC1v]
	outputs += [oFC1v]
	output_nms += ['oFC1v']
	
	# FC layer
	wFC2v = tf.Variable(tf.random_normal([N_FC1, 1], stddev=WEIGHT_STD), name='val')
	bFC2v = tf.Variable(tf.random_normal([1], mean=WEIGHT_STD*2, stddev=WEIGHT_STD), name='val')
	
	val = tf.tanh(tf.squeeze(tf.matmul(oFC1v, wFC2v) + bFC2v))

	weights += [wFC2v]
	outputs += [val]
	output_nms += ['val']

	# sq error
	val_mean_sq_err = tf.reduce_mean((val - val_target)**2)

	# pearson
	val_pearsonr = tf_pearsonr(val, val_target)

	########## FC l2 reg
	FC_L2_reg = 0
	for weights in [wFC1v, wFC2v, bFC1v, bFC2v, wFC1p, wFC2p, bFC1p, bFC2p]:
		FC_L2_reg += tf.reduce_sum(weights**2)
	FC_L2_reg *= (L2_LAMBDA/2.)

	################### movement from tree statistics
	visit_count_map = tf.placeholder(tf.float32, shape=(gv.BATCH_SZ, gv.map_szt))
	
	tree_prob_visit_coord = tf_op.prob_to_coord(visit_count_map, dir_pre, dir_a)
	tree_det_visit_coord = tf.argmax(visit_count_map, 1, output_type=tf.int32)

	tree_det_move_unit = tf_op.move_unit(tree_det_visit_coord, moving_player)
	tree_prob_move_unit = tf_op.move_unit(tree_prob_visit_coord, moving_player)

	################### initialize

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	
	loss = LSQ_LAMBDA * pol_mean_sq_err + \
	       LSQ_REG_LAMBDA * pol_mean_sq_reg_err + \
	       POL_CROSS_ENTROP_LAMBDA * pol_cross_entrop_err + \
	       VAL_LAMBDA * val_mean_sq_err + \
	       VALR_LAMBDA * val_pearsonr + \
	       tf.losses.get_regularization_loss() + FC_L2_reg

	with tf.control_dependencies(update_ops):
		train_step = tf.train.MomentumOptimizer(EPS, MOMENTUM).minimize(loss)

	sess.run(tf.global_variables_initializer())

	# saving
	saver = tf.train.Saver()
