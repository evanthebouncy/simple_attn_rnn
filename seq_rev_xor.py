from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf
from data_gen import *
from tensorflow.python.ops import array_ops

batch_size = 30
input_length = 20
output_length = int(input_length / 2)

# encoding biRnn params
# the data_gen function gives you data with size of 4, so you can't change this.
input_size = 4
num_units = 30

# the decoding rnn 
decode_num_units = 30
decode_input_size = num_units * 2
# the output character dimension is also 4, you can't change this
output_size = 4

# the context weight nn
weight_nn_hidden = 100
weight_nn_input_size = decode_num_units + output_size + num_units * 2

sess = tf.Session()
with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)) as scope:

  # our cells
  cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=num_units, 
                           input_size=input_size)

  cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=num_units, 
                           input_size=input_size)

  initializer = tf.random_uniform_initializer(-0.01, 0.01)
  # sequence_length = tf.placeholder(tf.int64)
  sequence_length = tf.placeholder(tf.int64, [batch_size])
  cell_fw = tf.nn.rnn_cell.LSTMCell(
      num_units, input_size, initializer=initializer)
  cell_bw = tf.nn.rnn_cell.LSTMCell(
      num_units, input_size, initializer=initializer)

  # input and outputs
  input_seq = input_length * [tf.placeholder(tf.float32, shape=(batch_size, input_size))]
  output_seq = output_length * [tf.placeholder(tf.float32, shape=(batch_size, output_size))]

  birnn_outputs = tf.nn.bidirectional_rnn(
      cell_fw, cell_bw, input_seq, dtype=tf.float32,
      sequence_length=sequence_length)


  # construct a simple feed forward NN to compute the weights e_ij
  # first run it through a hidden layer
  w1 = tf.Variable(tf.random_normal([weight_nn_input_size, weight_nn_hidden], mean=1.1, stddev=0.035), name="w1")
  b1 = tf.Variable(tf.zeros([weight_nn_hidden]), name="b1")

  # then output a weight eij as a scalar
  w2 = tf.Variable(tf.random_normal([weight_nn_hidden, 1], mean=0.1, stddev=0.035), name="w2")
  b2 = tf.Variable(tf.zeros([1]), name="b2")

  cell_decode = tf.nn.rnn_cell.LSTMCell(
    num_units=decode_num_units, input_size=decode_input_size, num_proj=output_size, initializer=initializer)

  state_init_var = tf.Variable(tf.random_normal([1, decode_num_units+output_size]), name="state_init")
  state_init = tf.tile(state_init_var, [batch_size, 1])

  _seq_out = []
  cur_state = state_init

  for i in range(output_length):
    print ("Iteration i ", i)
    # compute context
    relu1s = [tf.nn.relu(tf.matmul(array_ops.concat(1, [birnn_outputs[j], state_init]), w1) + b1) for j in range(input_length)]
    contexts_weights = [tf.nn.sigmoid(tf.matmul(relu1s[j], w2) + b2) for j in range(input_length)]
    # join the weights together
    weight_vect = array_ops.concat(1, contexts_weights)
    weight_softmax = tf.nn.softmax(weight_vect)
    weight_softmax_tiled = tf.tile(tf.reshape(weight_softmax, [-1, input_length, 1]), [1, 1, num_units * 2])
    contexts_as_tensor = tf.reshape(array_ops.concat(1, birnn_outputs), [-1, input_length, num_units * 2])
    weighted_contexts = weight_softmax_tiled * contexts_as_tensor
    context = tf.reduce_sum(weighted_contexts, 1)
    if i > 0:
      scope.reuse_variables()
    cur_out, cur_state = cell_decode (context, cur_state)
    _seq_out_i = tf.nn.softmax(cur_out)
    _seq_out.append(_seq_out_i)


  seq_cross_entropys = []
  for i in range(output_length):
    _seq_out_i = _seq_out[i]
    seq_out_i = output_seq[i]
    # minimize cross entropy against the true label
    cross_entropy = -tf.reduce_sum(seq_out_i*tf.log(_seq_out_i))
    seq_cross_entropys.append(cross_entropy)

  total_cross_entropy = sum(seq_cross_entropys)

  # get gradients and clip them, use adaptive to prevent explosions
  tvars = tf.trainable_variables()
  grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(cross_entropy, tvars)]
  optimizer = tf.train.AdagradOptimizer(0.01)
  train_step = optimizer.apply_gradients(zip(grads, tvars))




#  sess.run([tf.initialize_all_variables()])
#  print ("weight_softmax_tiled")
#  res = sess.run([train_step], feed_dict=feed_dict)
#  for r in res:
#    print(r)

  # =============== TRAINING AND TESTING ================== #
  # initialize
  sess.run([tf.initialize_all_variables()])
  # run over many epochs
  for j in range(50001):
    # get data dynamically from my data generator
    seq_in, seq_out, seq_len = gen_data_batch(batch_size, input_length)
    feed_dict = {}
    for i in range(input_length):
      feed_dict[input_seq[i]] = seq_in[i]
    for i in range(output_length):
      feed_dict[output_seq[i]] = seq_out[i]
    feed_dict[sequence_length] = seq_len

    # do evaluation every 100 epochs
    if (j % 100 == 0):
      print("total cross entropy on batch ", j)
      res = sess.run([total_cross_entropy], feed_dict=feed_dict)
      for r in res:
        print (r)
#      print("====current accuracy==== at epoch ", i)
#      correct_prediction = tf.equal(tf.argmax(output_label,1), tf.argmax(output_label_,1))
#      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#      res = sess.run([accuracy], feed_dict={input_seq: pos_data, output_label_: pos_label})
#      print("pos accuracy: ", res)
#      res = sess.run([accuracy], feed_dict={input_seq: neg_data, output_label_: neg_label})
#      print("neg accuracy: ", res)
    # otherwise do training
    else:
      sess.run(train_step, feed_dict=feed_dict)

