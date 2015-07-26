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
import collections
import math


DATA_PROCESSED = False

def getNextBatch(batch_size, n_seq, max_seq_len, input_dim, data_store_path):
    """
    This function extracts the next batch for training. The first half of
    the batches are random sequences(trips) from the same driver, while the
    second half of the data are random trips from random, different drivers.

    
    Parameters
    ----------
    batch_size: number of sequences in batch
    n_seq: total: number of possible sequences
    max_seq_len: length of the longest sequence
    input_dim: dimensionality of the data
    data_store_path: full path to the netCDF file (ends with .nc)
    
    Notes
    ---------
    Output batch will be of size: batch_size x max_seq_len x input_dim
    """
    
    rootgrp = Dataset(data_store_path)
    # create batch_size/2 + 1 random drivers. 1 for intra set and the rest for inter.
    random_drivers = np.sort(np.random.choice(n_drivers, size=1+batch_size/2, replace=False))
    # create batch_size/2 random sequences (trips) for the same driver
    random_sequences_inter = np.sort(np.random.choice(n_seq, size=batch_size/2, replace=False))
    # create batch_size/2 random sequences (trips) for different drivers specified in random_drivers
    random_sequences_intra = np.sort(np.random.choice(n_seq, size=batch_size/2, replace=False))
    
    # instantiate data and mask variables with zeros
    batch_train = np.zeros([batch_size, max_seq_len, input_dim],dtype='float32')
    batch_mask = np.zeros([batch_size, max_seq_len],dtype='float32')
    
    # read first half of the data from .nc file directly
    batch_train[0:batch_size/2,:,:] = \
        rootgrp.variables['data'][random_drivers[0], (random_sequences_intra),:,:]
    batch_mask[0:batch_size/2,:] = \
        rootgrp.variables['mask'][random_drivers[0], (random_sequences_intra),:]
  
    # second half of the data must be read within a loop, because both driver and sequence 
    # are random and we don't want each random driver-sequence combination but just one.
    # Note: indexing in netCDF variables works differently from numpy
    for ind in range(batch_size/2):
        batch_train[batch_size/2 + ind,:,:] = \
            rootgrp.variables['data'][random_drivers[1+ind], random_sequences_inter[ind],:,:]
        batch_mask[batch_size/2 + ind,:] = \
            rootgrp.variables['mask'][random_drivers[1+ind], random_sequences_inter[ind],:]
    rootgrp.close()
    
    return batch_train, batch_mask


def genDataNC(data_read_path, data_store_path):
    """
    This function generates the data for the network from the .csv files
    and preprocesses them. Preprocessing involves transformation from 
    cartesian coordinates to polar coordinates, application of the first
    derivative and normalization such that the largest value is 1 and smallest
    value is -1. 
    The data is stored as a netCDF4 file.

    
    Parameters
    ----------
    data_read_path: path to a folder containing subfolders for each driver,
    which contain .csv files for all sequences
    
    data_store_path: full path to the netCDF file (ends with .nc), 
    this file shall not yet exist, it will be created.
    """
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
    mask = rootgrp.createVariable('mask','i1',('n_drivers','n_sequences','max_seqLen',))
    dictionary = rootgrp.createVariable('dictionary', 'i4', ('n_drivers',))
    # loop through all folders for each driver. 
    for driver_index, driverPath in enumerate(driverPaths):
        # create dictionary entry, because drivers are not enumerated continuously from 1-N        
        dictionary[driver_index] = int(driverPath.split('/')[-1])
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
                # only store difference between new and old coordinates --> reduces the range of values
                # variable data_polar_old stores the old value, initialize with zeros for every new .csv file
                x_old, y_old = np.float64(0.0), np.float64(0.0)
                for row in reader:
                    x, y = np.float64(row[0]), np.float64(row[1])
                    x_diff, y_diff = x-x_old, y-y_old
                    # transform from cartesian to polar coordinates,
                    # divide direction by 20 (~2x mean --> mean ~= 0.5) and angle by pi/2
                    data_polar_diff = np.array([np.sqrt(x_diff**2 + y_diff**2)/20, np.arctan(y_diff/x_diff)*2/math.pi])
                    # remove nans from 0/0 division
                    if np.isnan(data_polar_diff[1]):
                        data_polar_diff[1] = 0.0
                    tmp_data[sequence_index, int(reader.line_num)-1, :] = data_polar_diff 
                    tmp_mask[sequence_index, int(reader.line_num)-1] = 1  
                    # save the old values for the next iteration(line in .csv file)                   
                    x_old, y_old = x,y

        # write temporary variables into netCDF4 file and delete temp
        data[driver_index,:,:,:] = tmp_data
        mask[driver_index,:,:] = tmp_mask
        del(tmp_data)
        del(tmp_mask)
    rootgrp.close() 
    return n_drivers, n_sequences, max_seqLen, input_dim

    
class SumLayer(lasagne.layers.Layer):
    """
    Performs a sum over the axis 1.
    E.g. an input with NxMxP dimensions results in NxP output
    
    Parameters
    ----------
    incoming : the layers feeding into this layer
    """
    
    def __init__(self, incoming, **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
    def get_output_for(self, input, **kwargs):
        return input.sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]
        
        
        
class ElemwiseMultLayer(lasagne.layers.MergeLayer):
    """
    This layer performs an elementwise mult of its input layers.
    It requires all input layers to have the same output shape.
    It can be used to multiply the data layer with the mask layer

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal
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
    """
    Reads the data (unpickle) from a filename into model
    
    Parameters
    ----------
    model: theano variable to store the weights, after they are read
    
    filename: filename specifying the pickle file that stores the model weights
    """
    filename = os.path.join('./', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """
    Writes the data (pickle) from a model into a file
    
    Parameters
    ----------
    model: theano variable of the final network layer 
    (for which get_all_param_values can be used)
    
    filename: filename specifying the pickle file that stores the model weights
    """
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w') as f:
        pickle.dump(data, f)



def getPatternMeanAndVar(patterns):
    """
    Extracts mean and variance of several patterns stored in a matrix of size
    number of patterns x pattern dimension.
    Mean and variance are both calculated elementwise
    
    Parameters
    ----------
    patterns: matrix of patterns, 1.st dim = number of patterns, 2nd dim = pattern dim
    """
    
    pattern_mean = np.mean(patterns,axis=0)
    pattern_var = np.var(patterns,axis=0)
    return pattern_mean, pattern_var


def predictPattern(pattern, pattern_mean, pattern_var):
    """
    This function calculates the prediction for a given pattern. This is determined by
    the mean of all patterns from a given driver and the hee variance of these
    patterns. Note that the mean of the variance along the output-dimension of the pattern
    is used, whereas the pattern-mean is given elementwise.
    
    If the mean(along output-dimension) of the quadratic difference of the pattern
    and the pattern-mean is greater than the pattern variance (multiplicated with a factor),
    the pattern is classified as a 1. Otherwise 0.
    
    Parameters
    ----------
    pattern: pattern (vector)
    
    pattern_mean: elementwise mean of all patterns from a driver (vector)
    
    pattern_var: mean of elementwise variance of all patterns from a driver (skalar)
    
    """
    if ((pattern - pattern_mean)**2).mean() < pattern_var * CLASSIFICATION_FACTOR:
        return 1
    else:
        return 0
    

# ************* create logger *************************
logger = logging.getLogger('logger')
while logger.root.handlers:
    print 'removing handler from logger.root'
    logger.root.removeHandler(logger.root.handlers[0])
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('testlog.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch) # add streamHandler
    logger.addHandler(fh) # and fileHandler
    


# ************* loading the test data **********logger.info('Generate + Load data...')******
data_read_path, data_store_path, model_path = \
    '../data/telematics/drivers/', '../data/telematics/telematics_data.nc','../data/telematics/models/'
if not DATA_PROCESSED:
    n_drivers, n_seq, max_seq_len, input_dim = genDataNC(data_read_path, data_store_path)
else:
    print 'file for data and mask already exists, no need to process'
    n_drivers, n_seq, max_seq_len, input_dim = 2736, 200, 1799, 2

    
DIM_OUT = 20 # dimension of the telematics pattern.    
# ************* definitions  for test data ***********************

# training parameters
BATCH_SIZE = 100 # 50 sequences of same driver, 50 of different drivers
N_EPOCHS = 20000 # needs < 14h training on gtx 980
LEARNING_RATE = .001
GRAD_CLIP = False

# Network size parameters
N_LSTM_HIDDEN1 = 128
N_LSTM_HIDDEN2 = 64

# Factor that determines, by which factor a pattern must differ by the mean (quadratically),
# to be classified as 1 (= belong to driver).
CLASSIFICATION_FACTOR = 1.8


# input dimensions. no need for output mask, output sequence always length 1 in this net
input_var = T.tensor3('inputs', dtype=theano.config.floatX) # BATCH_SIZE x MAX_INPUT_SEQ_LEN x N_INPUT_FEATURES
input_mask = T.tensor3('inputs_mask', dtype=theano.config.floatX) # BATCH_SIZE x MAX_INPUT_SEQ_LEN x N_LSTM_HIDDEN2

# ******************** create model ***************************************
logger.info('generating model...')
# CTC-Model: Here, only 1 output layer(lin), no tanh-layer to combine with RNN Transducer
l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, max_seq_len, input_dim), input_var=input_var)
l_in_mask = lasagne.layers.InputLayer(shape=(BATCH_SIZE, max_seq_len, N_LSTM_HIDDEN2), input_var=input_mask)
#************************************* LSTM Layer 1 ******************************************
lstm_1 = lasagne.layers.LSTMLayer(
    incoming=l_in, num_units=N_LSTM_HIDDEN1, grad_clipping=GRAD_CLIP, backwards=False)
#************************************* LSTM Layer 2 ******************************************
lstm_2 = lasagne.layers.LSTMLayer(
    incoming=lstm_1, num_units=N_LSTM_HIDDEN2, grad_clipping=GRAD_CLIP, backwards=False)
#************************************* Output *************************************************
# sum over "time" --> sum over 2nd dimension (length of the sequence)  
l_data_mask_mult = ElemwiseMultLayer(incomings=(lstm_2, l_in_mask))
l_out_sum = SumLayer(incoming=l_data_mask_mult)
# apply softmax
l_out_softmax = lasagne.layers.NonlinearityLayer(l_out_sum, nonlinearity=lasagne.nonlinearities.softmax)
# finally a fully connected layer of sigmoid units.
l_out = lasagne.layers.DenseLayer(
    incoming=l_out_softmax, num_units=DIM_OUT, nonlinearity=lasagne.nonlinearities.sigmoid)


net_output = lasagne.layers.get_output(l_out)
loss = batch_variance_ratio(net_output)

all_params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.adadelta(
    loss, all_params, learning_rate=lasagne.utils.floatX(LEARNING_RATE))


train = theano.function([input_var, input_mask],
                        outputs=[net_output, loss],
                        updates=updates)
compute_loss = theano.function(inputs=[input_var, input_mask],
                               outputs=loss)

forward_pass = theano.function(inputs=[input_var, input_mask],
                               outputs=[net_output])



logger.info('Training...')
# create ring buffers of size 50 to compute the moving average of losses (low pass "spiky" loss)
buf10 = collections.deque(maxlen=10)
buf50 = collections.deque(maxlen=50)
for epoch in range(N_EPOCHS):
    start_time = time.time()
    batch_train, batch_mask = getNextBatch(BATCH_SIZE, n_seq, max_seq_len, input_dim, data_store_path)
    # transform batch_mask into corresponding form with same dim as batch_train --> repeat along hidden dim
    batch_mask = np.repeat(np.expand_dims(batch_mask,axis=2), N_LSTM_HIDDEN2, axis=2)      
    _, t_loss = train(batch_train, batch_mask)
    
    # calc moving average of loss
    buf10.append(t_loss)
    buf50.append(t_loss)
    ma_loss_10 = np.array(0, dtype='float32')  
    ma_loss_50 = np.array(0, dtype='float32')  
    for b in buf10:
        ma_loss_10 += b/buf10.maxlen
    for b in buf50:
        ma_loss_50 += b/buf50.maxlen  
        
    end_time = time.time()
    logger.info("Epoch {} took {}, loss = {}, moving average loss: 10 = {}. 50 = {}"\
        .format(epoch, end_time - start_time, t_loss, ma_loss_10, ma_loss_50))
    
    if epoch % 1000 == 0:
        write_model_data(l_out, model_path+'model_'+str(epoch)+'.pkl')



logger.info('Prediction + write to submission.csv...')
# calculate total number of runs with a whole batch that can be made for one driver
# note: n_seq assumed to be a multiple of BATCH_SIZE
rootgrp = Dataset(data_store_path)
n_runs_per_driver = n_seq/BATCH_SIZE
with open('submission.csv', 'wb') as writefile:
    writer = csv.writer(writefile, delimiter=',')
    writer.writerow(['driver_trip','prob'])
    # loop through all drivers
    for driver_index in range(len(rootgrp.dimensions['n_drivers'])):
        print 'driver ' + str(driver_index) + ' of ' + str(n_drivers)
        # retrieve the real driver number (stored in dictionary)
        driver_number = rootgrp.variables['dictionary'][driver_index]   
        # retrieve the whole data of a particular driver
        driver_data = rootgrp.variables['data'][driver_index,:,:,:]
        driver_mask = rootgrp.variables['mask'][driver_index,:,:]
        patterns = np.zeros([n_seq, DIM_OUT],dtype='float32')
        
        for run in range(n_runs_per_driver):
            # from whole driver data extract batch (network needs exact batch size)
            batch_data = driver_data[run*BATCH_SIZE:(run+1)*BATCH_SIZE,:,:]
            batch_mask = driver_mask[run*BATCH_SIZE:(run+1)*BATCH_SIZE,:]
            batch_mask = np.repeat(np.expand_dims(batch_mask,axis=2), N_LSTM_HIDDEN2, axis=2) 
            # calc and output-patterns from network
            patterns[run*BATCH_SIZE:(run+1)*BATCH_SIZE,:] = forward_pass(batch_train, batch_mask)[0]
        # calculate mean and variance of patterns along dimension of all sequences
        pattern_mean, pattern_var = getPatternMeanAndVar(patterns)
        # predict each sequence (trip) independently, by comparing with mean and variance
        for sequence_index in range(n_seq):
            pred = predictPattern(patterns[sequence_index,:], pattern_mean, pattern_var.mean())
            writer.writerow([str(driver_number)+'_'+str(sequence_index+1), str(pred)])           
rootgrp.close()




                
                