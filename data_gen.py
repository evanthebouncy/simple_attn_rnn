import random
import numpy as np

# generate a random binary string (input)
# also generate its inverse (output)
def data_gen(seq_l):
  inpu = [random.randint(0,1) for i in range(seq_l)]
  rev = [x for x in reversed(inpu)]
  outpu = []
  for i in range(len(rev) / 2):
    outpu.append((rev[2*i] + rev[2*i+1]) % 2)
  return inpu, outpu 

# end = 2, pad = 3
def pad_data(data, data_l):
  return data + [2] + [3 for i in range(0, data_l - 1 - len(data))]

def data_to_tup(raw_data):
  def pt_xform(x):
    ret = [0. for i in range(4)]
    ret[x] = 1.
    return ret
  return [pt_xform(x) for x in raw_data]

def gen_data_batch(batchsize, examplesize):
  half_examplesize = examplesize / 2
  seq_ins = []
  seq_outs = []
  for i in range(0, batchsize):
    seq_l = 2 * random.randint(half_examplesize/2, half_examplesize)
    seq_i, seq_o = data_gen(seq_l)
    seq_i_pad = data_to_tup(pad_data(seq_i, examplesize))
    seq_o_pad = data_to_tup(pad_data(seq_o, examplesize / 2))
    seq_ins.append(seq_i_pad)
    seq_outs.append(seq_o_pad)
  return np.array(seq_ins, np.float32), np.array(seq_outs, np.float32)

    
