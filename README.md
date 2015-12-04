# Simple Construction of Attention Model in TF

## TLDR:


## Motivation

In this paper:

http://arxiv.org/abs/1409.0473

The authors constructed an attention model, with a bi-directional RNN as the
attention context, which is capable of doing a certain sequence manipulation
tasks, namely, translation from English to French with very good results.


We want to duplicate the model presented in this paper in its simplist form
(i.e. use it for a simpler sequence manipulation task than machine translation)
for clarity and understanding. 

# Goal of this exercise:

1. Use a bi-directional RNN consisting with LSTM units to proccess the input sequence into attention context
2. Construct a soft-max layer modeling attention to select choice contexts during decoding
3. Use this attention context in conjunction of a decoding RNN, generate the output sequence
4. Training and Evaluation of the model

Everything is bare-bone minimum as I'm just trying to learn the TF library

## The Sequence Manipulation Task

We solve a contrived task of taking in a binary input sequence of even length,
reverse the sequence, then perform xor on every pairs of its elements

i.e. on input [0,1,0,0,1,0,1,1]

we first reverse it [1,1,0,1,0,0,1,0]

then for every 2 consequtive elements, perform an xor

[xor(1,1), xor(0,1), xor(0,0), xor(1,0)]

the output is this:

[0, 1, 0, 1]

We'll train a neural network to perform this function with the TensorFlow library.

## Data Generation

look in data_gen.py for data generation, in particular, we use a 1-hot encoding
to represent ( and ), the sequence is also 0 padded, because TF does not
support dynamic unrolling, to take into account of various different lengths. 

For example, if we have an unrolling of size 6, and our sequence is ( ) ( ), we'll get the following encoding 
    
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

Here, [0, 1, 0] denotes (, [0, 0, 1] denotes ), and [1, 0, 0] denotes the 0 padding

The output is also 1-hot encoded, [1, 0] denotes true, and [0, 1] denotes false

Best way is to load data_gen into the interactive python shell via
execfile('data_gen.py') and execute some of the functions, they're very self
explainatory.

## RNN Model

The RNN model is a simple one, it is unrolled a fixed number of times, say 6, and conceptually perform the following program:

    def matching_paren(sequence):
      state = init_state
      for i in range(0, 6):
        output, state = lstm(sequence[i], state)
      return post_proccess(state)

Here, init_state is the initial state for the lstm unit, which is a vector that
can be learned, lstm is a rnn unit from the tf.models.rnn module, and
post_proccess is a small neural network with a layer of relu and a soft-max
layer for outputing true/false. 

Thus, all the tunable parameters are:

1. The initial state
2. All the weights in the LSTM unit
3. All the weights in the post_process units

See matching.py for details.

## RNN training and evaluation

For training, I compute the softmax as the predicted label. The error is the
cross entropy between the prediction and the true label. For training I'm using
gradient clipping as the gradient can become NaN otherwise, and I'm using
AdaptiveGradient because why the f not (maybe other is better but I have not
gotten around to learn to use those).

For evaluation, I evaluate the performance once every 100 epochs.

## Results

1. In matching.py, we use a single layer lstm and the result isn't as good, can only go to length of 8

2. In matching2.py, we use a stacked lstm and the result is better, can go up to length of 30

## Remarks
Hope this is helpful! I'm still new to TF so I probably can't answer any questions on TF reliably. Direct all questions to the TF discussion group on google.

