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
import time
import sys
sys.path.append('../../CTC-Connectionist-Temporal-Classification-master_pbrakel')
import ctc_cost as ctc
from netCDF4 import Dataset


logger = be.createLogger()

# ************** load netCDF4 datasets ***************************
dataPath = '../data/'
trainDataset = Dataset(dataPath+'train.nc')
valDataset = Dataset(dataPath+'val.nc')

# ************* definitions  for test data ***********************
# training parameters
BATCH_SIZE = 50
N_EPOCHS = 100
LEARNING_RATE = 1e-5
EPOCH_SIZE = 100 
GRAD_CLIP = False # clip large gradients in order to prevent exploding gradients,
GRADIENT_STEPS = -1 # defines how many timesteps are used for error backpropagation. -1 = all

# Network size parameters
N_LSTM_HIDDEN_UNITS = [250, 250, 250] # 3 BLSTM layers, 3 forward, 3 backward, each 250 units
INPUT_DIM = len(trainDataset.dimensions['inputPattSize']) #123
OUTPUT_DIM = len(trainDataset.dimensions['numLabels']) # 62
MAX_OUTPUT_SEQ_LEN = len(trainDataset.dimensions['maxLabelLength']) # 75
MAX_INPUT_SEQ_LEN = len(trainDataset.dimensions['maxInputLength']) # 778
#*******************************************************************************************************




#*******************************************************************************************************
logger.info('generating model...')
# Need linear for training with pseudo_cost and Softmax for validation with cost
model_lin, model_soft, l_in, l_mask = be.genModel(
    batch_size=BATCH_SIZE, max_input_seq_len=MAX_INPUT_SEQ_LEN,input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, 
        gradient_steps=GRADIENT_STEPS, grad_clip=GRAD_CLIP, lstm_hidden_units=N_LSTM_HIDDEN_UNITS)

#filename = '../data/paramValues/params_3l_epoch_39.pkl'
#paramValues = be.loadParams(filename)
#lasagne.layers.set_all_param_values(model_lin, paramValues)

output_lin = lasagne.layers.get_output(model_lin) 
output_softmax = lasagne.layers.get_output(model_soft) 

Y = T.matrix('target', dtype=theano.config.floatX) #BATCH_SIZE x MAX_OUTPUT_SEQ_LEN
Y_mask = T.matrix('target_mask', dtype=theano.config.floatX) #BATCH_SIZE x MAX_OUTPUT_SEQ_LEN

# get all parameters used for training
all_params = lasagne.layers.get_all_params(model_lin, trainable=True) 

ctc_cost_train = ctc.pseudo_cost(y=Y.dimshuffle((1,0)), \
                       y_hat=output_lin.dimshuffle((1,0,2)), \
                       y_mask=Y_mask.dimshuffle((1,0)), \
                       y_hat_mask=(l_mask.input_var).dimshuffle((1,0)), \
                       skip_softmax=True).mean(dtype=theano.config.floatX)
                       
ctc_cost_monitor = ctc.cost(y=Y.dimshuffle((1,0)), \
                            y_hat=output_softmax.dimshuffle((1,0,2)), \
                            y_mask=Y_mask.dimshuffle((1,0)), \
                            y_hat_mask=(l_mask.input_var).dimshuffle((1,0))).mean(dtype=theano.config.floatX)                                
                                      
updates = lasagne.updates.adadelta(
    ctc_cost_train, all_params, learning_rate=lasagne.utils.floatX(LEARNING_RATE))   

    
logger.info('compiling functions...')
train = theano.function([l_in.input_var, Y, l_mask.input_var, Y_mask],
                        outputs=[output_softmax, ctc_cost_monitor],
                        updates=updates)
                        
compute_cost = theano.function(
                inputs=[l_in.input_var, Y, l_mask.input_var, Y_mask],
                outputs=ctc_cost_monitor)

forward_pass = theano.function(inputs=[l_in.input_var, l_mask.input_var],
                               outputs=[output_softmax])                
#*******************************************************************************************************

 



#*******************************************************************************************************
logger.info("calc initial cost and PER..")

x_val_batch, y_val_batch, x_val_mask, y_val_mask = be.makeRandomBatchesFromNetCDF(valDataset, BATCH_SIZE)

cost_val = np.array([compute_cost(x, y, x_mask, y_mask)[()] \
    for x, y, x_mask, y_mask
    in zip(x_val_batch, y_val_batch, x_val_mask, y_val_mask)]).mean()  

# feed batches of data and mask through net, then reshape to flatten dimensions num_batches x batch_size 
net_outputs = np.array([forward_pass(x, x_mask)[0] for x, x_mask in zip(x_val_batch, x_val_mask)]) 
sequence_probdist = net_outputs \
    .reshape([net_outputs.shape[0]*net_outputs.shape[1],net_outputs.shape[2],net_outputs.shape[3]]) 

# also reshape masks and target. --> flatten num_batches x batch_size    
x_val_masks = x_val_mask.reshape([x_val_mask.shape[0]*x_val_mask.shape[1],x_val_mask.shape[2]]) 
y_val_masks = y_val_mask.reshape([y_val_mask.shape[0]*y_val_mask.shape[1],y_val_mask.shape[2]]) 
tar = y_val_batch.reshape([y_val_batch.shape[0]*y_val_batch.shape[1],y_val_batch.shape[2]]) 

# decode each training datum sequentially.
# TODO: decode in batches           
decoded = [be.decodeSequence(sequence_probdist[i], x_val_masks[i], OUTPUT_DIM-1) for i in range(sequence_probdist.shape[0])]

# calculate PER for each training datum sequentially. 
# TODO: PER in batches
PERs = [be.calcPER(tar[i, y_val_masks[i,:]==1], decoded[i]) for i in range(tar.shape[0])]
logger.info("Initial cost(val) = {},  PER-mean = {}".format(cost_val, np.mean(PERs)))
#*******************************************************************************************************





#*******************************************************************************************************
logger.info('Training...')
for epoch in range(N_EPOCHS):  
    print "make new random batches"
    x_train_batch, y_train_batch, x_train_mask, y_train_mask = be.makeRandomBatchesFromNetCDF(trainDataset, BATCH_SIZE)      
    start_time_epoch = time.time()
    
    for counter, (x, y, x_mask, y_mask) in enumerate(zip(x_train_batch, y_train_batch, x_train_mask, y_train_mask)):
        start_time = time.time()
        _, c = train(x, y, x_mask, y_mask)
        end_time = time.time()
        print "batch " + str(counter) + " duration: " + str(end_time - start_time) + " cost " + str(c.mean())
     
    print "\n"
    # since we have a very small val set, validate over complete val set  
    #***************************************************************************************      
    cost_val = np.array([compute_cost(x, y, x_mask, y_mask)[()] \
        for x, y, x_mask, y_mask
        in zip(x_val_batch, y_val_batch, x_val_mask, y_val_mask)]).mean()  
    
    # feed batches of data and mask through net, then reshape to flatten dimensions num_batches x batch_size 
    net_outputs = np.array([forward_pass(x, x_mask)[0] for x, x_mask in zip(x_val_batch, x_val_mask)]) 
    sequence_probdist = net_outputs \
        .reshape([net_outputs.shape[0]*net_outputs.shape[1],net_outputs.shape[2],net_outputs.shape[3]]) 
    
    # also reshape masks and target. --> flatten num_batches x batch_size    
    x_val_masks = x_val_mask.reshape([x_val_mask.shape[0]*x_val_mask.shape[1],x_val_mask.shape[2]]) 
    tar = y_val_batch.reshape([y_val_batch.shape[0]*y_val_batch.shape[1],y_val_batch.shape[2]]) 
    
    # decode each training datum sequentially.
    # TODO: decode in batches           
    decoded = [be.decodeSequence(sequence_probdist[i], x_val_masks[i], OUTPUT_DIM-1) for i in range(sequence_probdist.shape[0])]
    
    # calculate PER for each training datum sequentially. 
    # TODO: PER in batches
    PERs = [be.calcPER(tar[i,y_val_masks[i,:]==1], decoded[i]) for i in range(tar.shape[0])]   
    #***************************************************************************************
   
    end_time_epoch = time.time()
    logger.info("Epoch {} took {}, cost(val) = {}, PER-mean = {}".format(
        epoch, end_time_epoch - start_time_epoch, cost_val, np.mean(PERs)))

    if ((epoch+1)%10 == 0):
        filename = '../data/paramValues/params_3l_cost_epoch_' + str(epoch) + '.pkl' 
        be.saveParams(model_lin, filename)













