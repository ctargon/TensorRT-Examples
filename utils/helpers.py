#/usr/bin/python

import tensorflow as tf
import re

TOWER_NAME = 'tower'

"""
	function: _activation_summary

	Helper to create summaries for activations.
	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.
	
	Args:
		x: Tensor
	Returns:
		nothing
"""
def _activation_summary(x):

	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity',
						tf.nn.zero_fraction(x))


"""
	function: _variable_on_cpu
	
	Helper to create a Variable stored on CPU memory.
	
	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable
	Returns:
		Variable Tensor
"""
def _variable_on_cpu(FLAGS, name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	
	return var



"""
	function: _variable_with_weight_decay
	
	Helper to create an initialized Variable with weight decay.
	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.
	
	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.
	Returns:
		Variable Tensor
"""
def _variable_with_weight_decay(FLAGS, name, shape, stddev, wd):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	
	var = _variable_on_cpu(FLAGS, 
							name,
							shape,
							tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	
	return var









