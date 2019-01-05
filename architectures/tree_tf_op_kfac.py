import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
import global_vars as gv
import os
import kfac
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
		POL_CROSS_ENTROP_LAMBDA, VAL_LAMBDA, WEIGHT_STD=1e-2):
	
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
	layer_collection = kfac.LayerCollection()
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
	
	layer = tf.layers.Conv2D(filters=N_FILTERS[0], kernel_size=[FILTER_SZS[0]]*2, 
		kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_STD),
		strides=[STRIDES[0]]*2, padding="same", activation=None, name='conv0')
	preactivations = layer(imgs)
	activations = tf.nn.relu(preactivations)

	layer_collection.register_conv2d((layer.kernel, layer.bias), (1,1,1,1), "SAME", imgs, preactivations)

	convs += [activations]

	for i in range(1, len(N_FILTERS)):
		layer = tf.layers.Conv2D(filters=N_FILTERS[i], kernel_size=[FILTER_SZS[i]]*2,
				kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_STD),
				strides=[STRIDES[i]]*2, padding="same", activation=None, name='conv%i' % i)
		
		preactivations = layer(convs[i-1])

		layer_collection.register_conv2d((layer.kernel, layer.bias), (1,1,1,1), "SAME", convs[i-1], preactivations)

		# residual bypass
		if (i % 2) == 0:
			preactivations += convs[i-2]

		activations = tf.nn.relu(preactivations)
		convs += [activations]

	out_sz = np.int(np.prod(convs[-1].shape[1:]))
	convr = tf.reshape(convs[-1], [gv.BATCH_SZ, out_sz])

	################### pol
	# FC layer
	layer = tf.layers.Dense(N_FC1, kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_STD), name='FC1')
	preactivations = layer(convr)
	oFC1p = tf.nn.relu(preactivations)

	layer_collection.register_fully_connected((layer.kernel, layer.bias), convr, preactivations)

	# FC layer
	layer = tf.layers.Dense(map_prod, kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_STD), name='FC2')
	preactivations = layer(oFC1p)
	pol_pre = tf.nn.relu(preactivations)

	layer_collection.register_fully_connected((layer.kernel, layer.bias), oFC1p, preactivations)

	layer_collection.register_categorical_predictive_distribution(pol_pre)
	
	pol = tf.nn.softmax(pol_pre)
		
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

	global oFC1v, preactivations
	################# val
	# FC layer
	layer = tf.layers.Dense(N_FC1, kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_STD), name='v_FC1')
	preactivations = layer(convr)
	oFC1v = preactivations

	layer_collection.register_fully_connected((layer.kernel, layer.bias), convr, preactivations)

	# FC layer
	layer = tf.layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_STD), name='v_FC2')
	preactivations = layer(oFC1v)
	val = tf.squeeze(tf.tanh(preactivations))

	layer_collection.register_fully_connected((layer.kernel, layer.bias), oFC1v, preactivations)

	layer_collection.register_normal_predictive_distribution(val, var=1.0)

	# sq error
	val_mean_sq_err = tf.reduce_mean((val - val_target)**2)

	# pearson
	val_pearsonr = tf_pearsonr(val, val_target)

	
	################### movement from tree statistics
	visit_count_map = tf.placeholder(tf.float32, shape=(gv.BATCH_SZ, gv.map_szt))
	
	tree_prob_visit_coord = tf_op.prob_to_coord(visit_count_map, dir_pre, dir_a)
	tree_det_visit_coord = tf.argmax(visit_count_map, 1, output_type=tf.int32)

	tree_det_move_unit = tf_op.move_unit(tree_det_visit_coord, moving_player)
	tree_prob_move_unit = tf_op.move_unit(tree_prob_visit_coord, moving_player)

	################### initialize

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	
	loss = POL_CROSS_ENTROP_LAMBDA * pol_cross_entrop_err + \
	       VAL_LAMBDA * val_mean_sq_err

	params = tf.trainable_variables()
	grads = tf.gradients(loss, params)
        grad_params = list(zip(grads, params))

	learning_rate = .25
	damping_lambda = .01
	moving_avg_decay=.99
	kfac_norm_constraint = .0001
	kfac_momentum = .9

	optimizer = kfac.optimizer.KfacOptimizer(layer_collection=layer_collection, damping=damping_lambda,
		learning_rate=EPS, cov_ema_decay=moving_avg_decay,
		momentum=kfac_momentum, norm_constraint=kfac_norm_constraint)

	train_step = optimizer.apply_gradients(grad_params)


	#with tf.control_dependencies(update_ops):
	#	#train_step = tf.train.MomentumOptimizer(EPS, MOMENTUM).minimize(loss)
	#	train_step = tf.train.GradientDescentOptimizer(EPS).minimize(loss)

	sess.run(tf.global_variables_initializer())

	# saving
	saver = tf.train.Saver()
