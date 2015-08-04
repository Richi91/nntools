# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:44:06 2015

@author: richi-ubuntu
"""

import lasagne
import numpy as np
import theano
import theano.tensor as T
import backEnd as be
import frontEnd as fe
import time
import sys
sys.path.append('../../Lasagne-CTC')
import ctc_cost as ctc
import pickle as pkl


logger = be.createLogger()
# ************* loming the test data ****************
logger.info('Loading data...')
with open('../data/timit_data.pkl', 'rb') as input:
    X_train = pkl.load(input)
    X_val = pkl.load(input)
    X_test = pkl.load(input)
    y_train = pkl.load(input)
    y_val = pkl.load(input)
    y_test = pkl.load(input)

# ************* definitions  for test data ***********************
# training parameters
BATCH_SIZE = 50
N_EPOCHS = 50
LEARNING_RATE = 1e-5
EPOCH_SIZE = 100 
GRAD_CLIP = 100 # clip large gradients in order to prevent exploding gradients

# Network size parameters
INPUT_DIM = 123
OUTPUT_DIM = 62
LSTM_HIDDEN_UNITS = [250, 250, 250]
MAX_INPUT_SEQ_LEN = fe.getLongestSequence(X_train,X_val)
MAX_OUTPUT_SEQ_LEN = fe.getLongestSequence(y_train,y_val)

logger.info('generating batches...')
# *********** Convert to batches of time series of uniform length ************
# TODO: This should be done in training loop, because then random batches can be used. 
x_train_batch, x_train_mask, y_train_batch, y_train_mask = \
    be.makeBatches(X_train, y_train, MAX_INPUT_SEQ_LEN, MAX_OUTPUT_SEQ_LEN, BATCH_SIZE)
x_val_batch, x_val_mask, y_val_batch, y_val_mask = \
    be.makeBatches(X_val, y_val, MAX_INPUT_SEQ_LEN, MAX_OUTPUT_SEQ_LEN, BATCH_SIZE)

# ******************** create model ***************************************
logger.info('generating model...')
# input dimensions 
X = T.tensor3('X', dtype=theano.config.floatX) # BATCH_SIZE x MAX_INPUT_SEQ_LEN x INPUT_DIM
X_mask = T.matrix('X_mask', dtype=theano.config.floatX) # BATCH_SIZE x MAX_INPUT_SEQ_LEN

Y = T.matrix('Y', dtype=theano.config.floatX) #BATCH_SIZE x MAX_OUTPUT_SEQ_LEN
Y_mask = T.matrix('Y_mask', dtype=theano.config.floatX) #BATCH_SIZE x MAX_OUTPUT_SEQ_LEN

# model with linear and softmax output. Use linear for training with CTC and Softmax for validation or not at all..
model_lin, model_soft = be.genModel(batch_size=BATCH_SIZE, max_input_seq_len=MAX_INPUT_SEQ_LEN,input_dim=INPUT_DIM, 
    output_dim=OUTPUT_DIM, grad_clip=GRAD_CLIP, lstm_hidden_units=LSTM_HIDDEN_UNITS)

output_lin = lasagne.layers.get_output(model_lin, X, mask=X_mask)
output_softmax = lasagne.layers.get_output(model_soft, X, mask=X_mask) 

# get all parameters used for training
all_params = lasagne.layers.get_all_params(model_lin, trainable=True)

# the CTC cross entropy between y and linear output network
#pseudo_cost = ctc.pseudo_cost(y=Y.dimshuffle([1,0]), y_hat=output_lin.dimshuffle([1,0,2]), 
#   y_mask=Y_mask.dimshuffle([1,0]), y_hat_mask=X_mask.dimshuffle([1,0]),skip_softmax=True).mean()
pseudo_cost = ctc.pseudo_cost(y=Y, y_hat=output_lin, y_mask=Y_mask, mask=X_mask)
pseudo_cost_grad = T.grad(pseudo_cost.sum() / BATCH_SIZE, all_params)
updates = lasagne.updates.momentum(
    pseudo_cost_grad, all_params, learning_rate=lasagne.utils.floatX(LEARNING_RATE))   
    
logger.info('compiling functions...')

train = theano.function([X, Y, X_mask, Y_mask],
                        outputs=[output_lin, output_softmax, pseudo_cost.sum()],
                        updates=updates)
                        
compute_cost = theano.function(
                inputs=[X, Y, X_mask, Y_mask],
                outputs=[pseudo_cost.sum()])

forward_pass = theano.function(inputs=[X, X_mask],
                               outputs=[output_softmax])                


logger.info('Training...')
cost_val = np.array([compute_cost(x, y, x_mask, y_mask) \
    for x, y, x_mask, y_mask
    in zip(x_val_batch, y_val_batch, x_val_mask, y_val_mask)]).mean()
logger.info("Initial cost(val) = {}".format(cost_val))

for epoch in range(N_EPOCHS):        
    start_time_epoch = time.time()
    #shuffle all batches + loop through all batches
    batch_shuffle = np.random.choice(x_train_batch.shape[0], x_train_batch.shape[0], False)
    for x, x_mask, y, y_mask in zip(x_train_batch[batch_shuffle],
                                    x_train_mask[batch_shuffle],
                                    y_train_batch[batch_shuffle],
                                    y_train_mask[batch_shuffle]):
        start_time = time.time()
        sequence_shuffle = np.random.choice(x.shape[0], x.shape[0], False)
        # train: x,y,y_mask,y_hat_mask(=x_mask)
        _,_, c = train(x[sequence_shuffle], y[sequence_shuffle], x_mask[sequence_shuffle], y_mask[sequence_shuffle])
        end_time = time.time()
        print "batch duration:" + str(end_time - start_time) + " cost " + str(c.mean())
        #check_dimension(x, y, x_mask, y_mask)
    
    # small val set --> validate over complete val set  
    cost_val = np.array([compute_cost(x, y, x_mask, y_mask) \
                for x, y, x_mask, y_mask
                in zip(x_val_batch, y_val_batch, x_val_mask, y_val_mask)]).mean()
    # calculate phoneme error rates (PER)           
    net_outputs = np.array([forward_pass(x,x_mask)[0] for x,x_mask in zip(x_val_batch,x_val_mask)]) 
    sequence_probdist = net_outputs \
        .reshape([net_outputs.shape[0]*net_outputs.shape[1],net_outputs.shape[2],net_outputs.shape[3]]) 
    masks = x_val_mask.reshape([x_val_mask.shape[0]*x_val_mask.shape[1],x_val_mask.shape[2]])            
    decoded = [be.decodeSequence(sequence_probdist[i], masks[i], OUTPUT_DIM-1) for i in range(sequence_probdist.shape[0])]
    tar = y_val_batch.reshape([y_val_batch.shape[0]*y_val_batch.shape[1],y_val_batch.shape[2]])
    tar_mask = y_val_mask.reshape([y_val_mask.shape[0]*y_val_mask.shape[1],y_val_mask.shape[2]])   
    PERs = [be.calcPER(tar[i,tar_mask[i,:]==1], decoded[i]) for i in range(tar.shape[0])]
        
    end_time_epoch = time.time()
    logger.info("Epoch {} took {}, cost(val) = {}, PER-mean = {}".format(
        epoch, end_time_epoch - start_time_epoch, cost_val, np.mean(PERs)))


















