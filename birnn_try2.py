from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf
from data_gen import *
from tensorflow.python.ops import array_ops

batch_size = 2
input_length = 5
output_length = input_length / 2

# encoding biRnn params
# the data_gen function gives you data with size of 4, so you can't change this.
input_size = 4
num_units = 3

# the decoding rnn 
decode_num_units = 3
decode_input_size = num_units * 2

# the context weight nn
weight_nn_hidden = 10
weight_nn_input_size = decode_num_units * 2 + num_units * 2
# weight_nn_input_size = num_units * 2



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

  # input_seq = tf.placeholder(tf.float32, [input_length, None, input_size])
  input_seq = input_length * [tf.placeholder(tf.float32, shape=(batch_size, input_size))]


  birnn_outputs = tf.nn.bidirectional_rnn(
      cell_fw, cell_bw, input_seq, dtype=tf.float32,
      sequence_length=sequence_length)

  state_init_var = tf.Variable(tf.random_normal([1, decode_num_units * 2]))
  state_init = tf.tile(state_init_var, [batch_size, 1])

  # construct a simple feed forward NN to compute the weights e_ij
  # first run it through a hidden layer
  w1 = tf.Variable(tf.random_normal([weight_nn_input_size, weight_nn_hidden], mean=1.1, stddev=0.035))
  b1 = tf.Variable(tf.zeros([weight_nn_hidden]))
  # relu1s = [tf.nn.relu(tf.matmul(birnn_outputs[j], w1) + b1) for j in range(input_length)]
  relu1s = [tf.nn.relu(tf.matmul(array_ops.concat(1, [birnn_outputs[j], state_init]), w1) + b1) for j in range(input_length)]

  # then output a weight eij as a scalar
  w2 = tf.Variable(tf.random_normal([weight_nn_hidden, 1], mean=0.1, stddev=0.035))
  b2 = tf.Variable(tf.zeros([1]))
  contexts_weights = [tf.nn.sigmoid(tf.matmul(relu1s[j], w2) + b2) for j in range(input_length)]

  # join the weights together
  weight_vect = array_ops.concat(1, contexts_weights)

  weight_softmax = tf.nn.softmax(weight_vect)

  # weight_softmax_tiled = tf.tile(weight_softmax, [1, num_units * 2])
  weight_softmax_tiled = tf.tile(tf.reshape(weight_softmax, [-1, input_length, 1]), [1, 1, num_units * 2])

  contexts_as_tensor = tf.reshape(array_ops.concat(1, birnn_outputs), [-1, input_length, num_units * 2])

  weighted_contexts = weight_softmax_tiled * contexts_as_tensor

  context = tf.reduce_sum(weighted_contexts, 1)

  cell_decode = tf.nn.rnn_cell.LSTMCell(
    num_units=decode_num_units, input_size=decode_input_size, initializer=initializer)

#  fake_input = tf.Variable(tf.random_normal([batch_size, decode_input_size]))
#  fake_state = tf.Variable(tf.random_normal([batch_size, decode_num_units * 2]))
#  print ("input dim ", fake_input.get_shape())
#  print ("state dim ", fake_state.get_shape())
#  print ("accepted input dim ", cell_decode.input_size)
#  print ("accepted state dim ", cell_decode.state_size)

  out1, st1 = cell_decode (context, state_init)

  

  seq_in, seq_out, seq_len = gen_data_batch(batch_size, input_length)

  sess.run([tf.initialize_all_variables()])

  feed_dict = {}
  for i in range(input_length):
    feed_dict[input_seq[i]] = seq_in[i]
  feed_dict[sequence_length] = seq_len

#  print ("weight_softmax")
#  res1 = sess.run(weight_softmax, feed_dict=feed_dict)
#  for r in res1:
#    print(r)

  print ("weight_softmax_tiled")
  res = sess.run([out1, st1], feed_dict=feed_dict)
  for r in res:
    print(r)

# results
# out1 = ....................................[0.67272633, 0.67272633]
# st1  = [0.68967271, 0.68967271, 0.68967271, 0.67272633, 0.67272633]
# out2 = ....................................[1.29300046, 1.29300046]
# st2  = [1.59910131, 1.59910131, 1.59910131, 1.29300046, 1.29300046]
# out3 = ....................................[1.46279204, 1.46279204]
# st3  = [2.57725525, 2.57725525, 2.57725525, 1.46279204, 1.46279204]

