import random
import numpy as np

# generate a random binary string (input)
# also generate its xored inverse (output)
def data_gen(seq_l):
  inpu = [random.randint(0,1) for i in range(seq_l)]
  rev = [x for x in reversed(inpu)]
  outpu = []
  for i in range(len(rev) / 2):
    outpu.append((rev[2*i] + rev[2*i+1]) % 2)
  return inpu, outpu 

# input size is 4
# 0, 1 are the content, 2 is end symbol, 3 is padding
def pad_data(data, data_l):
  return data + [2] + [3 for i in range(0, data_l - 1 - len(data))]

def data_to_tup(padded_data):
  def pt_xform(x):
    ret = [0. for i in range(4)]
    ret[x] = 1.
    return ret
  return [pt_xform(x) for x in padded_data]

def gen_data_batch(batchsize, input_length):
  half_input_length = input_length / 2
  _seq_ins = []
  _seq_outs = []
  seq_lens = []

  # to feed the correct data into the BiRNN, we need to have data in the form of a list
  # input_length x batch_size x input_size (the input_length is the length of the list)
  # the batch_size x input_size is a tensor

  for i in range(0, batchsize):
    seq_l = 2 * random.randint(half_input_length/2, half_input_length)
    seq_i, seq_o = data_gen(seq_l)
    seq_i_pad = data_to_tup(pad_data(seq_i, input_length))
    seq_o_pad = data_to_tup(pad_data(seq_o, half_input_length))
    _seq_ins.append(seq_i_pad)
    _seq_outs.append(seq_o_pad)
    seq_lens.append(seq_l)

  # reshaping to form the right data format
  seq_ins = []
  seq_outs = []
  for j in range(input_length):
    at_j = [_seq_in[j] for _seq_in in _seq_ins]
    at_j = np.array(at_j, np.float32)
    seq_ins.append(at_j)
  for j in range(half_input_length):
    at_j = [_seq_out[j] for _seq_out in _seq_outs]
    at_j = np.array(at_j, np.float32)
    seq_outs.append(at_j)

  return seq_ins, seq_outs, np.array(seq_lens)


