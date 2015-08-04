# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:57:49 2015

@author: richi-ubuntu
"""
from __future__ import print_function
import numpy as np
import theano
import lasagne
import logging


def BLSTM_Concat(*args, **kwargs):
    kwargs.pop('backwards', None)
    l_fwd = lasagne.layers.LSTMLayer(*args, backwards=False, **kwargs)
    l_bwd = lasagne.layers.LSTMLayer(*args, backwards=True, **kwargs)
    return lasagne.layers.ConcatLayer((l_fwd,l_bwd),axis=2)


def makeBatches(X, y, input_sequence_length, output_sequence_length, batch_size):
    '''
    Convert a list of matrices into batches of uniform length
    :parameters:
        - X : list of np.ndarray
            List of matrices
        - y: list of np.ndarray
            list of vectors
        - input_sequence_length : int
            Desired input sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
       - output_sequence_length : int
            Desired output sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.   
        - batch_size : int
            Mini-batch size
    :returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, input_sequence_length, n_features)
        - X_mask : np.ndarray
            Mask denoting whether to include each time step of each time series
            shape=(n_batches, batch_size, input_sequence_length)
        - y_batch : np.ndarray
            Tensor of time series batches,
            shape=(n_batches, batch_size,output_sequence_length)
            - y_batch : np.ndarray
            Mask denoting whether to include each time step of each phoneme output series,
            shape=(n_batches, batch_size,output_sequence_length)
    '''
    n_batches = len(X)//batch_size # division with ceil (no non-full batches)
    # n_batches x batch_sz x in_seq_length x feature_dim
    X_batch = np.zeros((n_batches, batch_size, input_sequence_length, X[0].shape[1]),
                       dtype=theano.config.floatX) 
    # n_batches x batch_sz x out_seq_length
    y_batch = np.zeros((n_batches, batch_size, output_sequence_length),
                       dtype=theano.config.floatX) 
              
    X_mask = np.zeros((n_batches, batch_size, input_sequence_length), dtype=theano.config.floatX)          
    y_mask = np.zeros((n_batches, batch_size, output_sequence_length), dtype=theano.config.floatX)
    
    for b in range(n_batches):
        for n in range(batch_size):
            # read sequence from raw list of sequences
            X_m = X[b*batch_size + n] # has shape in_seq_len x feat_dim
            # put in the sequence to according position in X_batch
            X_batch[b, n, :X_m.shape[0],:] = X_m
            # mask (mark) all elements of X_batch, that belong to sequence with 1,
            # sequences shorter than max-length sequence are padded with zeros
            # and marked with 0 in X_mask
            X_mask[b, n, :X_m.shape[0]] = 1.0
            
            # similar with y
            y_m = y[b*batch_size + n]
            y_batch[b, n, :y_m.shape[0]] = y_m
            y_mask[b, n, :y_m.shape[0]] = 1.0
    return X_batch, X_mask, y_batch, y_mask
    


def makeBatchesNoCTC(X, y, sequence_length, batch_size):
    '''
    '''
    n_batches = len(X)//batch_size # division with ceil (no non-full batches)
    # n_batches x batch_sz x in_seq_length x feature_dim
    X_batch = np.zeros((n_batches, batch_size, sequence_length, X[0].shape[1]),
                       dtype=theano.config.floatX) 
    # n_batches x batch_sz x out_seq_length
    y_batch = np.zeros((n_batches, batch_size, sequence_length),
                       dtype=theano.config.floatX) 
              
    mask = np.zeros((n_batches, batch_size, sequence_length), dtype=theano.config.floatX)          

    
    for b in range(n_batches):
        for n in range(batch_size):
            # read sequence from raw list of sequences
            X_m = X[b*batch_size + n] # has shape in_seq_len x feat_dim
            # put in the sequence to according position in X_batch
            X_batch[b, n, :X_m.shape[0],:] = X_m
            # mask (mark) all elements of X_batch, that belong to sequence with 1,
            # sequences shorter than max-length sequence are padded with zeros
            # and marked with 0 in X_mask
            mask[b, n, :X_m.shape[0]] = 1.0
            
            # similar with y
            y_m = y[b*batch_size + n]
            y_batch[b, n, :y_m.shape[0]] = y_m
    return X_batch, y_batch, mask
    
    
    
    
def genModel(batch_size, max_input_seq_len, input_dim, output_dim, grad_clip, lstm_hidden_units):
    """
    Creates a deep BLSTM model. 3 layers of BLSTM units, where BLSTM units consist of 2 LSTM units,
    one calculating outputs for the forward sequence and one for backward sequence (reversed). The
    forward and backward LSTMs are merged by concatenating (alternative is sum). 
    The output of the 3 BLSTM Layers is fed into a fully connected layer. 
    The "post-output" layer is a Softmax.
    This function outputs both the model for the linear and the softmax output
    """
#************************************* Input Layer ********************************************
    l_in = lasagne.layers.InputLayer(shape=(batch_size, max_input_seq_len, input_dim))
    
##************************************* BLSTM Layer 1 ******************************************
#    lstm_forward0 = lasagne.layers.LSTMLayer(incoming=l_in, num_units=lstm_hidden_units[0], 
#        grad_clipping=grad_clip, backwards=False)
#    lstm_backward0 = lasagne.layers.LSTMLayer(incoming=l_in, num_units=lstm_hidden_units[0], 
#        grad_clipping=grad_clip, backwards=True)
#    l_concat0 = lasagne.layers.ConcatLayer((lstm_forward0, lstm_backward0), axis=2)
#    
##************************************* BLSTM Layer 2 ******************************************
#    lstm_forward1 = lasagne.layers.LSTMLayer(incoming=l_concat0, num_units=lstm_hidden_units[1],
#        grad_clipping=grad_clip, backwards=False)
#    lstm_backward1 = lasagne.layers.LSTMLayer(incoming=l_concat0, num_units=lstm_hidden_units[1],
#        grad_clipping=grad_clip, backwards=True)
#    l_concat1 = lasagne.layers.ConcatLayer((lstm_forward1, lstm_backward1), axis=2)
#    
##************************************* BLSTM Layer 3 ******************************************
#    lstm_forward2 = lasagne.layers.LSTMLayer(incoming=l_concat1, num_units=lstm_hidden_units[2], 
#        grad_clipping=grad_clip, backwards=False)
#    lstm_backward2 = lasagne.layers.LSTMLayer(incoming=l_concat1, num_units=lstm_hidden_units[2], 
#        grad_clipping=grad_clip, backwards=True)                                         
#    l_concat2 = lasagne.layers.ConcatLayer((lstm_forward2, lstm_backward2), axis=2)
  

    blstm0 = BLSTM_Concat(incoming=l_in, num_units=lstm_hidden_units[0],grad_clipping=grad_clip, backwards=False)
    blstm1 = BLSTM_Concat(incoming=blstm0, num_units=lstm_hidden_units[1],grad_clipping=grad_clip, backwards=False)
    blstm2 = BLSTM_Concat(incoming=blstm1, num_units=lstm_hidden_units[2],grad_clipping=grad_clip, backwards=False)
# Need to reshape hidden LSTM layers --> Combine batch size and sequence length for the output layer 
# Otherwise, DenseLayer would treat sequence length as feature dimension        
    l_reshape2 = lasagne.layers.ReshapeLayer(
        blstm2, (batch_size*max_input_seq_len, lstm_hidden_units[2] + lstm_hidden_units[2]))
    l_out = lasagne.layers.DenseLayer(
        incoming=l_reshape2, num_units=output_dim, nonlinearity=lasagne.nonlinearities.identity)
    
#************************************ linear output ******************************************
    l_out_lin_shp = lasagne.layers.ReshapeLayer(
        l_out, (batch_size, max_input_seq_len, output_dim))
    
#************************************ Softmax output *****************************************
    l_out_softmax = lasagne.layers.NonlinearityLayer(
        l_out, nonlinearity=lasagne.nonlinearities.softmax)
    l_out_softmax_shp = lasagne.layers.ReshapeLayer(
        l_out_softmax, (batch_size, max_input_seq_len, output_dim))   
        
    return l_out_lin_shp, l_out_softmax_shp
    
    
    
    
def genModelTEST(batch_size, max_input_seq_len, input_dim, output_dim, grad_clip, lstm_hidden_units):
#************************************* Input Layer ********************************************
    l_in = lasagne.layers.InputLayer(shape=(batch_size, max_input_seq_len, input_dim))
    lstm_forward0 = lasagne.layers.LSTMLayer(incoming=l_in, num_units=500, 
        grad_clipping=grad_clip, backwards=False)
  
# Need to reshape hidden LSTM layers --> Combine batch size and sequence length for the output layer 
# Otherwise, DenseLayer would treat sequence length as feature dimension        
    l_reshape2 = lasagne.layers.ReshapeLayer(
        lstm_forward0, (batch_size*max_input_seq_len, 500))
    l_out = lasagne.layers.DenseLayer(
        incoming=l_reshape2, num_units=output_dim, nonlinearity=lasagne.nonlinearities.identity)
    
#************************************ linear output ******************************************
    l_out_lin_shp = lasagne.layers.ReshapeLayer(
        l_out, (batch_size, max_input_seq_len, output_dim))
    
#************************************ Softmax output *****************************************
    l_out_softmax = lasagne.layers.NonlinearityLayer(
        l_out, nonlinearity=lasagne.nonlinearities.softmax)
    l_out_softmax_shp = lasagne.layers.ReshapeLayer(
        l_out_softmax, (batch_size, max_input_seq_len, output_dim))   
        
    return l_out_lin_shp, l_out_softmax_shp    
    
def decodeSequence(sequence_probdist, mask, blanksymbol):
    """
    This function decodes the output sequence from the network, which has the same length
    as the input sequence. The target sequence is generally shorter, thus the decoded sequence
    will also be shorter.
    """
    # just for testing, brute-force take output with highest prob and then eleminate repeated labels+blanks
    mostProbSeq = sequence_probdist[mask==1].argmax(axis=1)
    reduced = np.array([seq for index, seq in enumerate(mostProbSeq) \
        if (seq != mostProbSeq[index-1] or index == 0) and seq != blanksymbol])
    return reduced
    

def decodeSequenceNoCTC(sequence_probdist, mask):
    return np.array(sequence_probdist[mask==1].argmax(axis=1))


def calcPERNoCTC(tar,out):
    return (tar!=out).mean()
    
def calcPER(tar, out):
    """
    This function calculates the Phoneme Error Rate (PER) of the decoded networks output
    sequence (out) and a target sequence (tar) with Levenshtein distance and dynamic programming.
    """
    # initialize dynammic programming matrix
    D = np.zeros((len(tar)+1)*(len(out)+1), dtype=np.uint16)
    D = D.reshape((len(tar)+1, len(out)+1))
    # fill border entries, horizontals with timesteps of decoded networks output
    # and vertical with timesteps of target sequence.
    for t in range(len(tar)+1):
        for o in range(len(out)+1):
            if t == 0:
                D[0][o] = o
            elif o == 0:
                D[t][0] = t
                
    # compute the distance by calculating each entry successively. 
    # 
    for t in range(1, len(tar)+1):
        for o in range(1, len(out)+1):
            if tar[t-1] == out[o-1]:
                D[t][o] = D[t-1][o-1]
            else:
                # part-distances are 1 for all 3 possible paths (diag,hor,vert). 
                # Each elem of distance matrix D represents the accumulated part-distances
                # to reach this location in the matrix. Thus the distance at location (t,o)
                # can be calculated from the already calculated distance one of the possible 
                # previous locations(t-1,o), (t-1,o-1) or (t,o-1) plus the distance to the
                # desired new location (t,o). Since we are interested only in the shortes path,
                # take the shortes (min)
                substitution = D[t-1][o-1] + 1 # diag path
                insertion    = D[t][o-1] + 1 # hor path
                deletion     = D[t-1][o] + 1 # vert path
                D[t][o] = min(substitution, insertion, deletion)
    # best distance is bottom right entry of Distance-Matrix D.
    return float(D[len(tar)][len(out)])/len(tar)



def beamsearch(cost, extra, initial, B, E):
	"""A breadth-first beam search.
	B = max number of options to keep,
	E = max cost difference between best and worst threads in beam.
	initial = [ starting positions ]
	extra = arbitrary information for cost function.
	cost = fn(state, extra) -> (total_cost, [next states], output_if_goal)
 
     THIS FUNCTION IS HERE JUST FOR COMPARISON; WILL NEED OWN IMPLEMENTATION OF BEAMSEARCH
	"""

	o = []
	B = max(B, len(initial))
	hlist = [ (0.0, tmp) for tmp in initial ]
	while len(hlist)>0:
		# print "Len(hlist)=", len(hlist), "len(o)=", len(o)
		hlist.sort()
		if len(hlist) > B:
			hlist = hlist[:B]
		# print "E=", hlist[0][0], " to ", hlist[0][0]+E
		hlist = filter(lambda q, e0=hlist[0][0], e=E: q[0]-e0<=e, hlist)
		# print "		after: Len(hlist)=", len(hlist)
		nlist = []
		while len(hlist) > 0:
			c, point = hlist.pop(0)
			newcost, nextsteps, is_goal = cost(point, extra)
			if is_goal:
				o.append((newcost, is_goal))
			for t in nextsteps:
				nlist.append((newcost, t))
		hlist = nlist
	o.sort()
	return o    
 



 
def createLogger():
    """
    MOVE THIS INTO OTHER FILE - UTIL OR SOMETHING SIMILAR...
    """
    logger = logging.getLogger('logger')
    while logger.root.handlers:
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
        
    return logger