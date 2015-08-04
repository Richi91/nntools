# -*- coding: utf-8 -*-
import frontEnd as fe
import logging
"""
Created on Sat Jul 18 13:47:33 2015

@author: richi-ubuntu
"""

'''
Specify parameters in 'Configurations' below, the filepath to timit root and the
filepath to store the extracted data
The script extracts training, test and validation input and target data and saves
it to timit_data.pkl (path specified below)
You will need PySoundFile to read TIMIT .wav files and python_speech_features
    
For PySoundFile see: 
    See: http://pysoundfile.readthedocs.org/en/0.7.0/ and
    https://github.com/bastibe/PySoundFile
For see python_speech_features see:
    http://python-speech-features.readthedocs.org/en/latest/ or
    https://github.com/jameslyons/python_speech_features
'''
# ********************************* Configurations ************************************   
# 41 filters, 
rootdir = '../../TIMIT/timit'
#storePath = '../data/timit_data.pkl'
winlen, winstep, nfilt, lowfreq, highfreq, preemph, winSzForDelta, samplerate = \
0.025,  0.01,   40,     200,     8000,     0.97,    2,             16000    
     
nfft = fe.nextpow2(samplerate*winlen) 
n_speaker_val = 50 
# *************************************************************************************
    
    
# *********************************** create logger ***********************************
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
# *************************************************************************************
    

# ******* get paths --> extract features from datasets --> normalize datasets *********    
logger.info('Extract paths from database and split train set into train+val...')
wav_train, wav_val, wav_test, phn_train, phn_val, phn_test = fe.getTrainValTestPaths(rootdir,n_speaker_val)
    
logger.info('Calc features from training, val and test data (given path) ...')
X_train = fe.getAllFeatures(wav_train, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq,preemph,winSzForDelta)
X_val = fe.getAllFeatures(wav_val, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq,preemph,winSzForDelta)
X_test = fe.getAllFeatures(wav_test, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq,preemph,winSzForDelta)
    
logger.info('calculate mean and var from training-set --> normalise training-, test- and val-set ...')
X_train, X_val, X_test, mean_vector, std_vect = fe.normaliseFeatureVectors(X_train, X_val, X_test)
# get targets
logger.info('Extract target Sequences...')

#labelSequence_train = fe.getTargets(phn_train)
#labelSequence_val = fe.getTargets(phn_val)
#labelSequence_test = fe.getTargets(phn_test)

labelSequence_train = fe.getTargetsEachTimestep(phn_train, [len(s) for s in X_train], winstep, samplerate, winlen)
labelSequence_val = fe.getTargetsEachTimestep(phn_val, [len(s) for s in X_val], winstep, samplerate, winlen)
labelSequence_test = fe.getTargetsEachTimestep(phn_test, [len(s) for s in X_test], winstep, samplerate, winlen)

phonemeDic = fe.getPhonemeDictionary()

logger.info('convert test and train targets into numbers ...')  
y_train = fe.convertPhoneSequences(labelSequence_train, phonemeDic)
y_val = fe.convertPhoneSequences(labelSequence_val, phonemeDic)
y_test = fe.convertPhoneSequences(labelSequence_test, phonemeDic)



#**************************************************************************************
#**************************** save datasets *******************************************
import netCDF4
from netCDF4 import Dataset
import numpy as np
#rootgrp = Dataset('train_1_speaker_other.nc', 'a')
numSeqs_train = len(X_train)
numSeqs_val = len(X_val)
numSeqs_test = len(X_test)
numTimesteps_train = sum(len(s) for s in X_train)
numTimesteps_val = sum(len(s) for s in X_val)
numTimesteps_test = sum(len(s) for s in X_test)
inputPattSize = 123
maxSeqTagLength = 100 # This will be the name of the files. path from timit to file
numLabels = 62
maxLabelLength = fe.getLongestSequence(y_train,y_val)


train = Dataset('train.nc', 'w', format='NETCDF4')
try:
    # create dimension
    train.createDimension('numSeqs', numSeqs_train)
    train.createDimension('numTimesteps', numTimesteps_train)
    train.createDimension('inputPattSize', inputPattSize)
    train.createDimension('numLabels', numLabels)
    train.createDimension('maxLabelLength', maxLabelLength)
    train.createDimension('maxSeqTagLength', maxSeqTagLength)
    # create variables   
    seqTags = train.createVariable('seqTags','S1',('numSeqs', 'maxSeqTagLength',))
    seqLengths = train.createVariable('seqLengths','i4',('numSeqs',))
    inputs = train.createVariable('inputs','f4',('numTimesteps','inputPattSize',))
    targetClasses = train.createVariable('targetClasses','i4',('numTimesteps',))
    
    # write into variables    
    index = 0
    for s in wav_train:
        tmp_str = s.split('train',1)[1]
        tmp_str = np.array(tmp_str.split(","))
        tmp_str = netCDF4.stringtochar(tmp_str)
        seqTags[index,0:tmp_str.shape[1]] = tmp_str
        index += 1
       
    index = 0
    for s in wav_train:
        tmp_str = s.split('train',1)[1]
        for index_char in range(len(tmp_str)):
            seqTags[index,index_char] = tmp_str[index_char]
        index += 1    
       
    seqLengths[:] = [len(s) for s in X_train]
        
    index = 0
    for s in X_train:
        inputs[index:index+len(s),:] = s
        index += len(s)
        
    index = 0     
    # targets for every timestep --> no CTC
    for s in y_train:
        targetClasses[index:index+len(s)] = s
        index += len(s) 

except:
    print '\n\nfail train\n\n'
    pass
train.close()




val = Dataset('val.nc', 'w', format='NETCDF4')
try:
    # create dimension
    val.createDimension('numSeqs', numSeqs_val)
    val.createDimension('numTimesteps', numTimesteps_val)
    val.createDimension('inputPattSize', inputPattSize)
    val.createDimension('numLabels', numLabels)
    val.createDimension('maxLabelLength', maxLabelLength)
    val.createDimension('maxSeqTagLength', maxSeqTagLength)
    # create variables   
    seqTags = val.createVariable('seqTags','S1',('numSeqs', 'maxSeqTagLength',))
    seqLengths = val.createVariable('seqLengths','i4',('numSeqs',))
    inputs = val.createVariable('inputs','f4',('numTimesteps','inputPattSize',))
    targetClasses = val.createVariable('targetClasses','i4',('numTimesteps',))
    
    # write into variables    
    index = 0
    for s in wav_val:
        tmp_str = s.split('train',1)[1]
        tmp_str = np.array(tmp_str.split(","))
        tmp_str = netCDF4.stringtochar(tmp_str)
        seqTags[index,0:tmp_str.shape[1]] = tmp_str
        index += 1
             
    seqLengths[:] = [len(s) for s in X_val]
    
    index = 0
    for s in X_val:
        inputs[index:index+len(s),:] = s
        index += len(s)
        
    index = 0 
    # targets for every timestep --> no CTC
    for s in y_val:
        targetClasses[index:index+len(s)] = s
        index += len(s) 

except:
    print '\n\nfail val\n\n'
    pass
val.close()



test = Dataset('test.nc', 'w', format='NETCDF4')
try:
    # create dimension
    test.createDimension('numSeqs', numSeqs_test)
    test.createDimension('numTimesteps', numTimesteps_test)
    test.createDimension('inputPattSize', inputPattSize)
    test.createDimension('numLabels', numLabels)
    test.createDimension('maxLabelLength', maxLabelLength)
    test.createDimension('maxSeqTagLength', maxSeqTagLength)
    # create variables   
    seqTags = test.createVariable('seqTags','S1',('numSeqs', 'maxSeqTagLength',))
    seqLengths = test.createVariable('seqLengths','i4',('numSeqs',))
    inputs = test.createVariable('inputs','f4',('numTimesteps','inputPattSize',))
    targetClasses = test.createVariable('targetClasses','i4',('numTimesteps',))
    
    # write into variables    
    index = 0
    for s in wav_test:
        tmp_str = s.split('test',1)[1]
        tmp_str = np.array(tmp_str.split(","))
        tmp_str = netCDF4.stringtochar(tmp_str)
        seqTags[index,0:tmp_str.shape[1]] = tmp_str
        index += 1
             
    seqLengths[:] = [len(s) for s in X_test]
        
    index = 0
    for s in X_test:
        inputs[index:index+len(s),:] = s
        index += len(s)
                
    index = 0       
    # targets for every timestep --> no CTC
    for s in y_test:
        targetClasses[index:index+len(s)] = s
        index += len(s) 

except:
    print '\n\nfail test\n\n'
    pass
test.close()




print 'finished'

    