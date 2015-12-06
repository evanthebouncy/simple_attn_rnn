from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf
from data_gen import *

batch_size = 1
# the data_gen function gives you data with size of 4, so you can't change this.
input_size = 4
num_units = 3

input_length = 8


sess = tf.Session()
with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)) as scope:

  # our cells
  cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=num_units, 
                           input_size=input_size)

  cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=num_units, 
                           input_size=input_size)

  initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=123)
  # sequence_length = tf.placeholder(tf.int64)
  sequence_length = tf.placeholder(tf.int64, [batch_size])
  cell_fw = tf.nn.rnn_cell.LSTMCell(
      num_units, input_size, initializer=initializer)
  cell_bw = tf.nn.rnn_cell.LSTMCell(
      num_units, input_size, initializer=initializer)

  # input_seq = tf.placeholder(tf.float32, [input_length, None, input_size])
  input_seq = input_length * [tf.placeholder(tf.float32, shape=(batch_size, input_size))]


  outputs = tf.nn.bidirectional_rnn(
      cell_fw, cell_bw, input_seq, dtype=tf.float32,
      sequence_length=sequence_length)

  # input_value = np.random.randn(batch_size, input_size)
  _inputs_values = [[1.0 for i in range(input_size)] for j in range(batch_size)]
  inputs_values = [np.array(_inputs_values, np.float32) for k in range(input_length)]
  # inputs_values = [np.random.randn(batch_size, input_size) for i in range(input_length)]
  seq_in, seq_out, seq_len = gen_data_batch(batch_size, input_length)

  sess.run([tf.initialize_all_variables()])

  feed_dict = {}
  for i in range(input_length):
    feed_dict[input_seq[i]] = seq_in[i]
  feed_dict[sequence_length] = seq_len

  res = sess.run(outputs, feed_dict=feed_dict)
  for r in res:
    print(r)

# results
# out1 = ....................................[0.67272633, 0.67272633]
# st1  = [0.68967271, 0.68967271, 0.68967271, 0.67272633, 0.67272633]
# out2 = ....................................[1.29300046, 1.29300046]
# st2  = [1.59910131, 1.59910131, 1.59910131, 1.29300046, 1.29300046]
# out3 = ....................................[1.46279204, 1.46279204]
# st3  = [2.57725525, 2.57725525, 2.57725525, 1.46279204, 1.46279204]

