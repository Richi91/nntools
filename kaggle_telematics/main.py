# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:44:06 2015

@author: richi-ubuntu
"""

import logging
import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
import csv
import glob
import pickle
import os
from netCDF4 import Dataset

DATA_PROCESSED = True

def getNextBatch(BATCH_SIZE, n_seq, max_seq_len, input_dim, data_store_path):
    
    rootgrp = Dataset(data_store_path)
    # create BATCH_SIZE/2 + 1 random drivers. 1 for intra set and the rest for inter.
    random_drivers = np.sort(np.random.choice(n_drivers, size=1+BATCH_SIZE/2, replace=False))
    # create BATCH_SIZE/2 random sequences (trips) for the same driver
    random_sequences_inter = np.sort(np.random.choice(n_seq, size=BATCH_SIZE/2, replace=False))
    # create BATCH_SIZE/2 random sequences (trips) for different drivers specified in random_drivers
    random_sequences_intra = np.sort(np.random.choice(n_seq, size=BATCH_SIZE/2, replace=False))
    
    batch_train = np.zeros([BATCH_SIZE, max_seq_len, input_dim],dtype='float32')
    batch_mask = np.zeros([BATCH_SIZE, max_seq_len],dtype='float32')
    # ([NUMBER_OF_DRIVERS, n_seq, max_seq_len, input_dim])
    
#    with open(data_store_path+'data_and_mask_' + str(random_drivers[0]) + '.pkl', 'rb') as input:
#        d = pkl.load(input)
#        m = pkl.load(input)        
#    batch_train[0:BATCH_SIZE/2,:,:] = d[(random_sequences_intra),:,:]
#    batch_mask[0:BATCH_SIZE/2,:,:] = m[(random_sequences_intra),:,:]
    
    # Have to load each .pkl for each driver separately. Iterate through random list "random_drivers"
    # For each of these files, choose the corresponding random number in "random_sequences_inter"
    batch_train[0:BATCH_SIZE/2,:,:] = rootgrp.variables['data'][random_drivers[0], (random_sequences_intra),:,:]
    batch_mask[0:BATCH_SIZE/2,:] = rootgrp.variables['mask'][random_drivers[0], (random_sequences_intra),:]
  
    #start_time = time.time()
    for ind in range(BATCH_SIZE/2):
        batch_train[BATCH_SIZE/2 + ind,:,:] = rootgrp.variables['data'][random_drivers[1+ind], random_sequences_inter[ind],:,:]
        batch_mask[BATCH_SIZE/2 + ind,:] = rootgrp.variables['mask'][random_drivers[1+ind], random_sequences_inter[ind],:]
    #end_time = time.time()
    #dt = end_time-start_time    
    #print dt
    rootgrp.close()
    
    return batch_train, batch_mask


def genDataNC(data_read_path, data_store_path):
    # get Paths for all folder with data for each driver
    driverPaths = glob.glob(data_read_path+'*')    
    # create Dataset in netcdf4-format
    rootgrp = Dataset(data_store_path, 'w', format='NETCDF4')
    rootgrp.close() 
    rootgrp = Dataset(data_store_path, 'a')
    # create dimensions of netCDF4 file
    n_drivers, n_sequences, max_seqLen, input_dim = len(driverPaths), 200, 1799, 2
    rootgrp.createDimension('n_drivers', len(driverPaths))
    rootgrp.createDimension('n_sequences', 200)
    rootgrp.createDimension('max_seqLen', 1799)
    rootgrp.createDimension('input_dim', 2)
    # create variables of netCDF4 file
    data = rootgrp.createVariable('data','f4',('n_drivers','n_sequences','max_seqLen','input_dim',))
    mask = rootgrp.createVariable('mask','i1',('n_drivers','n_sequences','max_seqLen'))

    # loop through all folders for each driver. 
    for driver_index, driverPath in enumerate(driverPaths):
        # temporarily store and fill 3D array in RAM for fast write,
        # then write as a whole into netCDF4-file
        tmp_data = np.zeros([n_sequences, max_seqLen, input_dim]) 
        tmp_mask = np.zeros([n_sequences, max_seqLen])
        # get all csv files with sequences (= trips) for the corresponding driver
        sequences = glob.glob(driverPath+'/*.csv')
        print 'driver ' + str(driver_index) + ' of ' + str(n_drivers)
        # loop through all .csv files and read the content into temporary variables (in RAM)
        for sequence_index, sequence in enumerate(sequences):
            with open(sequence, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
                reader.next() #skip header
                for row in reader:
                    tmp_data[sequence_index, int(reader.line_num)-1, :] = [float(row[0]),float(row[1])]
                    tmp_mask[sequence_index, int(reader.line_num)-1] = 1  
        # write temporary variables into netCDF4 file and delete temp
        data[driver_index,:,:,:] = tmp_data
        mask[driver_index,:,:] = tmp_mask
        del(tmp_data)
        del(tmp_mask)
    rootgrp.close() 
    return n_drivers, n_sequences, max_seqLen, input_dim

    
class SumLayer(lasagne.layers.Layer):
    """
    SumLayer(incoming, axis)
    
    Performs a sum over the axis 1.
    E.g. summing over axis 1 of NxMxP dimensions results in NxP output
    
    """
    def get_output_for(self, input, **kwargs):
        return input.sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]
        
        
        
class ElemwiseMultLayer(lasagne.layers.MergeLayer):
    """
    This layer performs an elementwise mult of its input layers.
    It requires all input layers to have the same output shape.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal

    coeffs: list or scalar
        A same-sized list of coefficients, or a single coefficient that
        is to be applied to all instances. By default, these will not
        be included in the learnable parameters of this layer.

    Notes
    -----
    Depending on your architecture, this can be used to avoid the more
    costly :class:`ConcatLayer`. For example, instead of concatenating layers
    before a :class:`DenseLayer`, insert separate :class:`DenseLayer` instances
    of the same number of output units and add them up afterwards. (This avoids
    the copy operations in concatenation, but splits up the dot product.)
    """
    def __init__(self, incomings, **kwargs):
        super(ElemwiseMultLayer, self).__init__(incomings, **kwargs)
        self.merge_function = theano.tensor.mul
        
    def get_output_shape_for(self, input_shapes):
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same: " + str(input_shapes))
        return input_shapes[0]        

    def get_output_for(self, inputs, **kwargs):
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output        




        
def batch_variance_ratio(y):
    """
    This function computes a loss function that aims to
    minimize the variance within a class and maximize the variance accross differnet classes.
    The samples in dimension 0 (BATCH_SIZE) must be 50% of the same class and 50% of different classes.
    Thus the first 50% represent intra-class data and the last 50% inter-class data.
    1. compute elementwise variance along the dimension of data samples for intra and inter class
    2. compute elementwise ratio var_intra/var_inter, elementwise means for each feature dimension independently (no cov)
    3. compute squared sum of these variances 
    """
    n = theano.tensor.shape(y)[0]
    var_intra = theano.tensor.var(y[0:n/2],axis=0)
    var_inter = theano.tensor.var(y[n/2:n],axis=0)
    return ((var_intra / var_inter)**2).sum()
        
    



def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w') as f:
        pickle.dump(data, f)



# ************* create logger *************************
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('testlog.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch) # add streamHandler
logger.addHandler(fh) # and fileHandler

# ************* loading the test data ****************
logger.info('ToDo: Generate + Load data...')
data_read_path, data_store_path, model_path = \
    '../data/telematics/drivers/', '../data/telematics/telematics_data.nc','../data/telematics/'
if not DATA_PROCESSED:
    n_drivers, n_seq, max_seq_len, input_dim = genDataNC(data_read_path, data_store_path)
else:
    print 'file for data and mask already exists'
    n_drivers, n_seq, max_seq_len, input_dim = 2736, 200, 1799, 2
    
DIM_OUT = 20 # dimension of the telematics pattern.    
# ************* definitions  for test data ***********************

# training parameters
BATCH_SIZE = 100 #This will produce 100 sequences of a driver of the same class and 100 sequences if different drivers 
N_EPOCHS = 20000 # needs ~1day training on gtx 980
LEARNING_RATE = .001
GRAD_CLIP = 100 ## TODO: check value

# Network size parameters
N_LSTM_HIDDEN1 = 150
N_LSTM_HIDDEN2 = 150


# input dimensions. no need for output mask, output sequence always length 1 in this net
input_var = T.tensor3('inputs', dtype=theano.config.floatX) # BATCH_SIZE x MAX_INPUT_SEQ_LEN x N_INPUT_FEATURES
input_mask = T.tensor3('inputs_mask', dtype=theano.config.floatX) # BATCH_SIZE x MAX_INPUT_SEQ_LEN x N_LSTM_HIDDEN2

# ******************** create model ***************************************
logger.info('generating model...')
# CTC-Model: Here, only 1 output layer(lin), no tanh-layer to combine with RNN Transducer
l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, max_seq_len, input_dim), input_var=input_var)
l_in_mask = lasagne.layers.InputLayer(shape=(BATCH_SIZE, max_seq_len, N_LSTM_HIDDEN2), input_var=input_mask)
#************************************* BLSTM Layer 1 ******************************************
lstm_forward1 = lasagne.layers.LSTMLayer(
    incoming=l_in, num_units=N_LSTM_HIDDEN1, grad_clipping=GRAD_CLIP, backwards=False)
#************************************* BLSTM Layer 2 ******************************************
lstm_forward2 = lasagne.layers.LSTMLayer(
    incoming=lstm_forward1, num_units=N_LSTM_HIDDEN2, grad_clipping=GRAD_CLIP, backwards=False)
#************************************* Output *************************************************
# sum over "time" --> sum over 2nd dimension (length of the sequence)  
l_data_mask_mult = ElemwiseMultLayer(incomings=(lstm_forward2, l_in_mask))
l_out_sum = SumLayer(incoming=l_data_mask_mult)
# apply softmax
l_out_softmax = lasagne.layers.NonlinearityLayer(l_out_sum, nonlinearity=lasagne.nonlinearities.softmax)
# finally a fully connected layer of sigmoid units.
l_out = lasagne.layers.DenseLayer(
    incoming=l_out_softmax, num_units=DIM_OUT, nonlinearity=lasagne.nonlinearities.sigmoid)


prediction = lasagne.layers.get_output(l_out)
loss = batch_variance_ratio(prediction)

all_params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.adadelta(
    loss, all_params, learning_rate=lasagne.utils.floatX(LEARNING_RATE))


train = theano.function([input_var, input_mask],
                        outputs=[prediction, loss],
                        updates=updates)
compute_loss = theano.function(inputs=[input_var, input_mask],
                               outputs=loss)

forward_pass = theano.function(inputs=[input_var, input_mask],
                               outputs=[prediction,loss])

logger.info('Training...')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    batch_train, batch_mask = getNextBatch(BATCH_SIZE, n_seq, max_seq_len, input_dim, data_store_path)
    batch_mask = np.repeat(np.expand_dims(batch_mask,axis=2), N_LSTM_HIDDEN2, axis=2)
    
    train(batch_train, batch_mask)

    l = compute_loss(batch_train, batch_mask)
    
    end_time = time.time()
    logger.info("Epoch {} took {}, loss of curr. batch = {}".format(epoch, end_time - start_time, l))

    if epoch % 1000:
        write_model_data(l_out, model_path+'model_'+str(epoch)+'.pkl')





