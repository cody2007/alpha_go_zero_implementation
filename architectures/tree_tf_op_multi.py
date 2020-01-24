import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
import global_vars as gv
import os

hdir = os.getenv('HOME')

imgs_shape = [gv.BATCH_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels]
map_prod = np.prod(gv.map_sz)

gm_var_nms = ['board', 'valid_mv_map_internal']
gm_var_placeholders = ['set_var_int8']*2

gm_vars = {}; set_gm_vars = {}

def return_vars():
	v = {}
	with tf.device(DEVICE):
		for var in gm_var_nms:
			exec('v["%s"] = sess.run(gm_vars["%s"])' % (var, var))
	return v

def set_vars(v):
	with tf.device(DEVICE):
		for var, placeholder in zip(gm_var_nms, gm_var_placeholders):
			exec('sess.run(set_gm_vars["%s"], feed_dict={%s: v["%s"].ravel()})' % (var, placeholder, var))

def tf_pearsonr(val, val_target_nmean):
	val_nmean = val - tf.reduce_mean(val)
	val_target_nmean = val_target - tf.reduce_mean(val_target)
	
	val_std = tf.sqrt(tf.reduce_sum(val_nmean**2))
	val_target_std = tf.sqrt(tf.reduce_sum(val_target_nmean**2))

	return -tf.reduce_sum(val_nmean * val_target_nmean) / (val_std * val_target_std)


# the `training` input dictates whether batch norm statistics are updated
def init_model(DEVICE, N_FILTERS, FILTER_SZS, STRIDES, N_FC1, EPS, MOMENTUM, \
		LSQ_LAMBDA, LSQ_REG_LAMBDA, POL_CROSS_ENTROP_LAMBDA, 
		VAL_LAMBDA, VALR_LAMBDA, L2_LAMBDA, WEIGHT_STD=1e-2, training=True):
	
	global sess, tf_op, set_var_int32, set_var_int8	
	global convs, weights, output_nms, pol, pol_pre, pol_mean_sq_err, train_step
	global val, val_mean_sq_err, pol_loss, entrop, saver, update_ops 
	global val_pearsonr, pol_mean_sq_reg_err, loss, Q_map, P_map, visit_count_map
	global move_random_ai, init_state, nn_move_unit, nn_prob_move_unit
        global tree_to_coords, nn_max_to_coords, nn_prob_to_coords
	global tree_prob_move_unit, backup_visit, backup_visit_terminal, tree_det_move_unit
	global nn_prob_move_unit, nn_max_move_unit, tree_prob_visit_coord, tree_det_visit_coord
	global sess, imgs, imgs32, valid_mv_map, pol_target, val_target, moving_player
	global gm_vars, set_gm_vars, oFC1, session_restore, session_backup
	global winner, to_coords_input
	global score, n_captures, pol_cross_entrop_err
	global nn_prob_to_coords_valid_mvs, nn_max_prob_to_coords_valid_mvs
	global nn_prob_move_unit_valid_mvs, nn_max_prob_move_unit_valid_mvs
	global move_frm_inputs
	assert len(N_FILTERS) == len(FILTER_SZS) == len(STRIDES)

	imgs = {}; valid_mv_map = {}
	move_random_ai = {}

	convs = {}; weights = {}; output_nms = {}
	pol = {}; pol_pre = {}; val = {}
	nn_max_to_coords = {}; nn_prob_to_coords = {}; nn_prob_to_coords_valid_mvs = {}
	nn_max_prob_to_coords_valid_mvs = {}
	nn_max_move_unit = {}; nn_prob_move_unit = {}; nn_prob_move_unit_valid_mvs = {}
	nn_max_prob_move_unit_valid_mvs = {}

	with tf.device(DEVICE):
		sess = tf.InteractiveSession()
		if DEVICE == '/gpu:0':
			tf_op = tf.load_op_library('cuda_op_kernel_75.so')
		else:
			tf_op = tf.load_op_library('cuda_op_kernel_52.so')

		##################### set / load vars
		set_var_int32 = tf.placeholder(tf.int32, shape=[None])
		set_var_int8 = tf.placeholder(tf.int8, shape=[None])

		#### init state
		init_state = tf_op.init_state()

		moving_player = tf.placeholder(tf.int8, shape=())
		winner, score, n_captures = tf_op.return_winner(moving_player)

		visit_count_map = tf.placeholder(tf.float16, shape=(gv.BATCH_SZ, gv.map_szt)) # map of visits
		to_coords_input = tf.placeholder(tf.int16, shape=gv.BATCH_SZ) # simply the coordinates
		
		pol_target = tf.placeholder(tf.float32, shape=[gv.BATCH_SZ, map_prod])
		val_target = tf.placeholder(tf.float32, shape=[gv.BATCH_SZ])

		session_restore = tf_op.session_restore()
		session_backup = tf_op.session_backup()

		##### vars
		for var, placeholder in zip(gm_var_nms, gm_var_placeholders):
			exec('gm_vars["%s"] = tf_op.%s()' % (var, var))
			exec('set_gm_vars["%s"] = tf_op.set_%s(%s)' % (var, var, placeholder))

		#### imgs
		imgs, valid_mv_map = tf_op.create_batch(moving_player) # output is float16
		imgs32 = tf.cast(imgs, tf.float32)
		assert imgs.shape == tf.placeholder(tf.float16, shape=imgs_shape).shape, 'tf op shape not matching global_vars'
		move_random_ai = tf_op.move_random_ai(moving_player)

		move_frm_inputs = tf_op.move_unit(to_coords_input, moving_player) # deterministically move from input coordinates

		################### movement from tree statistics (must be supplied--these are placeholders)
		tree_prob_visit_coord = tf_op.prob_to_coord(visit_count_map)
		tree_det_visit_coord = tf.cast(tf.argmax(visit_count_map, 1, output_type=tf.int32), tf.int16)
		
		tree_det_move_unit = tf_op.move_unit(tree_det_visit_coord, moving_player)
		tree_prob_move_unit = tf_op.move_unit(tree_prob_visit_coord, moving_player)
		
		############ specifics of how 3 networks will be initialized (on each card)
		scopes = ['eval', 'main', 'eval32']
		dtypes = ['float16', 'float16', 'float32']
		if training:
			#trainings = [False, False, True]
			trainings = [True, True, True]
		else:
			trainings = [False, False, False]
		
		for s in scopes:
			convs[s] = []; weights[s] = []; output_nms[s] = []

		################ network (f32 and f16 weights)
		for s, d, t in zip(scopes, dtypes, trainings):
			with tf.variable_scope(s):
				if s == 'eval32':
					# conv2d: "channels_last (default) corresponds to inputs with shape (batch, height, width, channels)"
					# https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/layers/Conv2D
					
					# batch_norm: "Can be used as a normalizer function for conv2d and fully_connected. The normalization
					#              is over all but the last dimension if data_format is NHWC (default)"
					# https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/contrib/layers/batch_norm
					convs[s] += [tf.nn.relu(tf.contrib.layers.batch_norm(tf.layers.conv2d(inputs=imgs32, filters=N_FILTERS[0], kernel_size=[FILTER_SZS[0]]*2, 
						strides=[STRIDES[0]]*2, padding="same", activation=None, name='conv0',
						kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA),
						bias_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA)), is_training=t))]
				else: # float16 models
					convs[s] += [tf.nn.relu(tf.contrib.layers.batch_norm(tf.layers.conv2d(inputs=imgs, filters=N_FILTERS[0], kernel_size=[FILTER_SZS[0]]*2, 
						strides=[STRIDES[0]]*2, padding="same", activation=None, name='conv0',
						kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA),
						bias_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA)), is_training=t))]

				weights[s] += [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, s + '/conv0')[0]]
				output_nms[s] += ['conv0']
				
				# convolutional layers
				for i in range(1, len(N_FILTERS)):
					output_nms[s] += ['conv' + str(i)]
					
					conv_out = tf.contrib.layers.batch_norm(\
						tf.layers.conv2d(inputs=convs[s][i-1], filters=N_FILTERS[i], kernel_size=[FILTER_SZS[i]]*2,
							strides=[STRIDES[i]]*2, padding="same", activation=None, name=output_nms[s][-1],
							kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA),
							bias_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_LAMBDA)), is_training=t)
					
					# residual bypass
					if (i % 2) == 0:
						conv_out += convs[s][i-2]

					convs[s] += [tf.nn.relu(conv_out)]

					weights[s] += [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, s + '/' + output_nms[s][-1])[0]]

				out_sz = np.int(np.prod(convs[s][-1].shape[1:]))
				convr = tf.reshape(convs[s][-1], [gv.BATCH_SZ, out_sz])

				################### policy output head (pol)
				# FC layer
				wFC1p = tf.Variable(tf.random_normal([out_sz, N_FC1], stddev=WEIGHT_STD, dtype=d), name='wFC1')
				bFC1p = tf.Variable(tf.random_normal([N_FC1], mean=WEIGHT_STD*2, stddev=WEIGHT_STD, dtype=d), name='bFC1')
				
				oFC1p = tf.nn.relu(tf.matmul(convr, wFC1p) + bFC1p)

				weights[s] += [wFC1p]
				output_nms[s] += ['oFC1p']

				# FC layer
				wFC2p = tf.Variable(tf.random_normal([N_FC1, map_prod], stddev=WEIGHT_STD, dtype=d), name='wFC2')
				bFC2p = tf.Variable(tf.random_normal([map_prod], mean=WEIGHT_STD*2, stddev=WEIGHT_STD, dtype=d), name='bFC2')
				
				pol_pre[s] = tf.nn.relu(tf.matmul(oFC1p, wFC2p) + bFC2p)

				weights[s] += [wFC2p]
				output_nms[s] += ['pol_pre']
				
				pol[s] = tf.nn.softmax(pol_pre[s])
				output_nms[s] += ['pol']
				
				#if s != 'eval32':
				nn_max_to_coords[s] = tf.cast(tf.argmax(pol_pre[s], 1, output_type=tf.int32), 'int16')
				if s == 'eval32':
					pol16 = tf.cast(pol[s], tf.float16)
					nn_prob_to_coords[s] = tf_op.prob_to_coord(pol16) 
					nn_prob_to_coords_valid_mvs[s] = tf_op.prob_to_coord_valid_mvs(pol16)
					nn_max_prob_to_coords_valid_mvs[s] = tf_op.max_prob_to_coord_valid_mvs(pol16)

				else:
					nn_prob_to_coords[s] = tf_op.prob_to_coord(pol[s]) 
					nn_prob_to_coords_valid_mvs[s] = tf_op.prob_to_coord_valid_mvs(pol[s])
					nn_max_prob_to_coords_valid_mvs[s] = tf_op.max_prob_to_coord_valid_mvs(pol[s])

				####### move unit
				# (these take as input coordinates and return flags indicating if movement was possible for each game)
				nn_max_move_unit[s] = tf_op.move_unit(nn_max_to_coords[s], moving_player)
				nn_prob_move_unit[s] = tf_op.move_unit(nn_prob_to_coords[s], moving_player)
				nn_prob_move_unit_valid_mvs[s] = tf_op.move_unit(nn_prob_to_coords_valid_mvs[s], moving_player)
				nn_max_prob_move_unit_valid_mvs[s] = tf_op.move_unit(nn_max_prob_to_coords_valid_mvs[s], moving_player)
				
				################# value output head (val)
				# FC layer
				wFC1v = tf.Variable(tf.random_normal([out_sz, N_FC1], stddev=WEIGHT_STD, dtype=d), name='val_wFC1')
				bFC1v = tf.Variable(tf.random_normal([N_FC1], mean=WEIGHT_STD*2, stddev=WEIGHT_STD, dtype=d), name='val_bFC1')
				
				#oFC1v = tf.nn.relu(tf.matmul(convr, wFC1v) + bFC1v)
				oFC1v = tf.matmul(convr, wFC1v) + bFC1v

				weights[s] += [wFC1v]
				output_nms[s] += ['oFC1v']
				
				# FC layer
				wFC2v = tf.Variable(tf.random_normal([N_FC1, 1], stddev=WEIGHT_STD, dtype=d), name='val')
				bFC2v = tf.Variable(tf.random_normal([1], mean=WEIGHT_STD*2, stddev=WEIGHT_STD, dtype=d), name='val')
				
				val[s] = tf.tanh(tf.squeeze(tf.matmul(oFC1v, wFC2v) + bFC2v))

				weights[s] += [wFC2v]
				output_nms[s] += ['val']

				
				################### initialize loss
				if s == 'eval32':
					########## FC l2 reg
					FC_L2_reg = 0
					for t_weights in [wFC1v, wFC2v, bFC1v, bFC2v, wFC1p, wFC2p, bFC1p, bFC2p]:
						FC_L2_reg += tf.reduce_sum(t_weights**2)
					FC_L2_reg *= (L2_LAMBDA/2.)

					##### pol
					# sq
					sq_err = tf.reduce_sum((pol[s] - pol_target)**2, axis=1)
					pol_mean_sq_err = tf.reduce_mean(sq_err)

					# sq reg
					sq_err_reg = tf.reduce_sum(pol_pre[s]**2, axis=1)
					pol_mean_sq_reg_err = tf.reduce_mean(sq_err_reg)

					# cross entrop
					pol_ln = tf.log(pol[s])
					pol_cross_entrop_err = -tf.reduce_mean(pol_target*pol_ln)
					
					#### val
					# sq error
					val_mean_sq_err = tf.reduce_mean((val[s] - val_target)**2)

					# pearson
					val_pearsonr = tf_pearsonr(val[s], val_target)

					update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=s)
					
					loss = LSQ_LAMBDA * pol_mean_sq_err + \
					       LSQ_REG_LAMBDA * pol_mean_sq_reg_err + \
					       POL_CROSS_ENTROP_LAMBDA * pol_cross_entrop_err + \
					       VAL_LAMBDA * val_mean_sq_err + \
					       VALR_LAMBDA * val_pearsonr + \
					       tf.losses.get_regularization_loss(s) + FC_L2_reg

					with tf.control_dependencies(update_ops):
						train_step = tf.train.MomentumOptimizer(EPS, MOMENTUM).minimize(loss)

				sess.run(tf.global_variables_initializer())

	# saving
	saver = tf.train.Saver()
