import tensorflow as tf
import numpy as np
import time
import os
import copy
import random
import global_vars as gv
from scipy.stats import pearsonr
import kfac

use_kfac = False
EPS = 1e-4
BATCH_SZ = 128
IMG_SZ = 9
N = 64#32#128
N_FILTERS = [N, N, N, N]
FILTER_SZS = [3,3, 3, 3]
STRIDES = [1,1, 1, 1]
N_FC1 = 64#32#128
MOMENTUM = .9

if use_kfac:
	save_nm = 'vars.npy'
else:
	save_nm = 'vars_wo_kfac.npy'

sess = tf.InteractiveSession()
layer_collection = kfac.LayerCollection()

imgs = tf.placeholder(tf.float32, shape=[BATCH_SZ, IMG_SZ, IMG_SZ, 1])
val_target = tf.placeholder(tf.float32, shape=[gv.BATCH_SZ])
pol_target = tf.placeholder(tf.float32, shape=[gv.BATCH_SZ, 9])

convs = []

###
layer = tf.layers.Conv2D(filters=N_FILTERS[0], kernel_size=[FILTER_SZS[0]]*2, 
	strides=[STRIDES[0]]*2, padding="same", activation=None, name='conv0')
preactivations = layer(imgs)
activations = tf.nn.relu(preactivations)

layer_collection.register_conv2d((layer.kernel, layer.bias), (1,1,1,1), "SAME", imgs, preactivations)

convs += [activations]

###
for i in range(1, len(N_FILTERS)):
	layer = tf.layers.Conv2D(filters=N_FILTERS[i], kernel_size=[FILTER_SZS[i]]*2,
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

################ pol
# FC layer
layer = tf.layers.Dense(N_FC1, kernel_initializer=tf.random_normal_initializer(), name='FC1')
preactivations = layer(convr)
oFC1p = tf.nn.relu(preactivations)

layer_collection.register_fully_connected((layer.kernel, layer.bias), convr, preactivations)

# FC layer
layer = tf.layers.Dense(9, kernel_initializer=tf.random_normal_initializer(), name='FC2')
preactivations = layer(oFC1p)
pol_pre = tf.nn.relu(preactivations)

layer_collection.register_fully_connected((layer.kernel, layer.bias), oFC1p, preactivations)

layer_collection.register_categorical_predictive_distribution(pol_pre)

pol = tf.nn.softmax(pol_pre)
pol_ln = tf.log(pol)
pol_cross_entrop_err = -tf.reduce_mean(pol_target*pol_ln)

################# val
# FC layer
layer = tf.layers.Dense(N_FC1, kernel_initializer=tf.random_normal_initializer(), name='v_FC1')
preactivations = layer(convr)
oFC1v = preactivations

layer_collection.register_fully_connected((layer.kernel, layer.bias), convr, preactivations)

# FC layer
layer = tf.layers.Dense(1, kernel_initializer=tf.random_normal_initializer(), name='v_FC2')
preactivations = layer(oFC1v)
val = tf.squeeze(tf.tanh(preactivations))

layer_collection.register_fully_connected((layer.kernel, layer.bias), oFC1v, preactivations)

layer_collection.register_normal_predictive_distribution(val, var=1.0)

# sq error
val_mean_sq_err = tf.reduce_mean((val - val_target)**2)

###
print 'a'
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

loss = val_mean_sq_err + 5e-3*pol_cross_entrop_err

params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grad_params = list(zip(grads, params))

learning_rate = .25
damping_lambda = .01
moving_avg_decay=.99
kfac_norm_constraint = .0001
kfac_momentum = .9

if use_kfac:
	print 'b'
	optimizer = kfac.optimizer.KfacOptimizer(layer_collection=layer_collection, damping=damping_lambda,
		learning_rate=learning_rate, cov_ema_decay=moving_avg_decay,
		momentum=kfac_momentum, norm_constraint=kfac_norm_constraint)
	print 'c'
	train_step = optimizer.apply_gradients(grad_params)
	print 'd'
else:
	with tf.control_dependencies(update_ops):
		train_step = tf.train.MomentumOptimizer(EPS, MOMENTUM).minimize(loss)

sess.run(tf.global_variables_initializer())

print 'e'

val_err = []
pol_err = []

##########################
for i in range(20000):
	inputs = np.random.random(size=((imgs.shape))) - .5
	t = np.random.randint(2, size=BATCH_SZ)
	l = t*2 - 1
	inputs[:,0,0,0] += l
	pol_t = np.zeros((BATCH_SZ, 9), dtype='single')
	pol_t[t == 0] = np.arange(9)[np.newaxis]
	pol_t[t == 1] = np.arange(9)[::-1][np.newaxis]

	d = {imgs: inputs, val_target: t, pol_target: pol_t}
	val_err_tmp, v, pol_err_tmp, p = sess.run([val_mean_sq_err, val, pol_cross_entrop_err, pol, train_step], feed_dict=d)[:-1]
	val_err.append(val_err_tmp)
	pol_err.append(pol_err_tmp)
	if i % 100 == 0:
		print i, val_err_tmp, pol_err_tmp, v.std(), p.std()
		np.save('/tmp/' + save_nm, {'val_err': val_err, 'pol_err': pol_err})
