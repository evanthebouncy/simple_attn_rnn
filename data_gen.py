import random
import numpy as np

# generate a random binary string (input)
# also generate its inverse (output)
def data_gen(data_l):
  inpu = [random.randint(0,1) for i in range(data_l)]
  rev = [x for x in reversed(inpu)]
  outpu = []
  for i in range(len(rev) / 2):
    outpu.append((rev[2*i] + rev[2*i+1]) % 2)
  return inpu, outpu 

def pad_data(data, data_l):
  return data + [0 for i in range(0, data_l - len(data))]

def data_to_tup(raw_data):
  def pt_xform(x):
    if x == 0:
      return [1., 0., 0.]
    if x == 1:
      return [0., 1., 0.]
    if x == 2:
      return [0., 0., 1.]
  return [pt_xform(x) for x in raw_data]

def gen_data_batch(batchsize, examplesize, pos_neg = None):
  dataz = []
  labelz = []
  for i in range(0, batchsize):
    label_i = random.random() > 0.5
    if pos_neg == True:
      label_i = True
    if pos_neg == False:
      label_i = False
    data_i = single_pos_data_gen(examplesize) if label_i else single_neg_data_gen(examplesize)
    dataz.append(data_to_tup(pad_data(data_i, examplesize)))
    labelz.append([1., 0.] if label_i else [0., 1.])
  return np.array(dataz, np.float32), np.array(labelz, np.float32)

    
