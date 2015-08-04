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
import pickle as pkl


logger = be.createLogger()
# ************* loading the test data ****************
logger.info('Loading data...')
with open('../data/timit_data_noCTC.pkl', 'rb') as input:
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
OUTPUT_DIM = 61
#LSTM_HIDDEN_UNITS = [250, 250, 250] # will not be used in genModelTEST
SEQ_LEN = fe.getLongestSequence(X_train,X_val)



logger.info('generating batches...')
# *********** Convert to batches of time series of uniform length ************
# TODO: This should be done in training loop, because then random batches can be used. 
x_train_batch, y_train_batch, train_mask = \
    be.makeBatchesNoCTC(X_train, y_train, SEQ_LEN, BATCH_SIZE)
x_val_batch, y_val_batch, val_mask = \
    be.makeBatchesNoCTC(X_val, y_val, SEQ_LEN, BATCH_SIZE)


# ******************** create model ***************************************
logger.info('generating model...')
# model with linear and softmax output. Use linear for training with CTC and Softmax eventually for validation
_, model_soft, l_in, l_mask  = be.genModelTEST(batch_size=BATCH_SIZE, max_input_seq_len=SEQ_LEN,input_dim=INPUT_DIM, 
    output_dim=OUTPUT_DIM, grad_clip=GRAD_CLIP)

output_softmax = lasagne.layers.get_output(model_soft) 

output_flat = T.reshape(output_softmax, [BATCH_SIZE*SEQ_LEN, OUTPUT_DIM])

Y = T.matrix('target', dtype=theano.config.floatX) #BATCH_SIZE x SEQ_LEN
cost = T.mean(T.nnet.categorical_crossentropy(output_flat,
                                       T.cast(T.reshape(Y,[BATCH_SIZE*SEQ_LEN]), 'int32')))

# get all parameters used for training
all_params = lasagne.layers.get_all_params(model_soft) 
                                      
updates = lasagne.updates.momentum(
    cost, all_params, learning_rate=lasagne.utils.floatX(LEARNING_RATE))   
    
logger.info('compiling functions...')
train = theano.function([l_in.input_var, Y, l_mask.input_var],
                        outputs=[output_softmax, cost],
                        updates=updates)
                        
compute_cost = theano.function(
                inputs=[l_in.input_var, Y, l_mask.input_var],
                outputs=cost)


forward_pass = theano.function(inputs=[l_in.input_var, l_mask.input_var],
                               outputs=[output_softmax])                


#***************************************************************************************  
logger.info("calc initial cost and PER..")
cost_val = [compute_cost(x, y, mask) \
    for x, y, mask
    in zip(x_val_batch, y_val_batch, val_mask)][0].mean()  

# feed batches of data and mask through net, then reshape to flatten dimensions num_batches x batch_size 
net_outputs = np.array([forward_pass(x,mask)[0] for x, mask in zip(x_val_batch,val_mask)]) 
sequence_probdist = net_outputs \
    .reshape([net_outputs.shape[0]*net_outputs.shape[1],net_outputs.shape[2],net_outputs.shape[3]]) 

# also reshape masks and target. --> flatten num_batches x batch_size    
masks = val_mask.reshape([val_mask.shape[0]*val_mask.shape[1],val_mask.shape[2]]) 
tar = y_val_batch.reshape([y_val_batch.shape[0]*y_val_batch.shape[1],y_val_batch.shape[2]]) 

# decode each training datum sequentially.
# TODO: decode in batches           
decoded = [be.decodeSequenceNoCTC(sequence_probdist[i], masks[i]) for i in range(sequence_probdist.shape[0])]

# calculate PER for each training datum sequentially. 
# TODO: PER in batches
PERs = [be.calcPERNoCTC(tar[i,masks[i,:]==1], decoded[i]) for i in range(tar.shape[0])]
logger.info("Initial cost(val) = {},  PER-mean = {}".format(cost_val, np.mean(PERs)))
#***************************************************************************************

logger.info('Training...')
for epoch in range(N_EPOCHS):        
    start_time_epoch = time.time()
    #shuffle all batches + loop through all batches
    batch_shuffle = np.random.choice(x_train_batch.shape[0], x_train_batch.shape[0], False)
    counter = 0
    for x, y, xy_mask in zip(x_train_batch[batch_shuffle],
                                    y_train_batch[batch_shuffle],
                                    train_mask[batch_shuffle]):
        counter+=1
        start_time = time.time()
        sequence_shuffle = np.random.choice(x.shape[0], x.shape[0], False)
        # train: x,y,y_mask,y_hat_mask(=x_mask)
        _, c = train(x[sequence_shuffle], y[sequence_shuffle], xy_mask[sequence_shuffle])
        end_time = time.time()
        print "batch duration:" + str(end_time - start_time) + " cost " + str(c.mean())
  
    # since we have a very small val set, validate over complete val set  
    print "\n"
    
    #***************************************************************************************      
    cost_val = [compute_cost(x, y, mask) \
        for x, y, mask
        in zip(x_val_batch, y_val_batch, val_mask)][0].mean()  
    
    # feed batches of data and mask through net, then reshape to flatten dimensions num_batches x batch_size 
    net_outputs = np.array([forward_pass(x,mask)[0] for x, mask in zip(x_val_batch,val_mask)]) 
    sequence_probdist = net_outputs \
        .reshape([net_outputs.shape[0]*net_outputs.shape[1],net_outputs.shape[2],net_outputs.shape[3]]) 
    
    # also reshape masks and target. --> flatten num_batches x batch_size    
    masks = val_mask.reshape([val_mask.shape[0]*val_mask.shape[1],val_mask.shape[2]]) 
    tar = y_val_batch.reshape([y_val_batch.shape[0]*y_val_batch.shape[1],y_val_batch.shape[2]]) 
    
    # decode each training datum sequentially.
    # TODO: decode in batches           
    decoded = [be.decodeSequenceNoCTC(sequence_probdist[i], masks[i]) for i in range(sequence_probdist.shape[0])]
    
    # calculate PER for each training datum sequentially. 
    # TODO: PER in batches
    PERs = [be.calcPERNoCTC(tar[i,masks[i,:]==1], decoded[i]) for i in range(tar.shape[0])]   
    #***************************************************************************************
   
    end_time_epoch = time.time()
    logger.info("Epoch {} took {}, cost(val) = {}, PER-mean = {}".format(
        epoch, end_time_epoch - start_time_epoch, cost_val, np.mean(PERs)))


















