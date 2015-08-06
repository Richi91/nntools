This is an attempt to implementing deep BLSTM networks with Lasagne (fork of recurrent branch, see https://github.com/craffel/nntools/tree/recurrent/lasagne).
The network is applied to phoneme recognition on the TIMIT dataset. 

Still under strong development!...

requirements: 
	- theano 
	- netCDF + netCDF4-Python to read and store data (same is used in CURRENNT library)
	- PySoundFile to read timit's depcrecated .wav-like format, See: http://pysoundfile.readthedocs.org/en/0.7.0/ and https://github.com/bastibe/PySoundFile
	- python_speech_features for preprocessing (FFT-based filterbank), see http://python-speech-features.readthedocs.org/en/latest/ and https://github.com/jameslyons/python_speech_features


- Use frontEnd.py' function CreateNetCDF(default when running frontEnd) to generate netCDF4 data file with preprocessed timit data. You will need to adjust timit root path and data store path.
- Run testNoCTC.py for training a deep BLSTM on preprocessed TIMIT data without CTC, thus on each frame. Need HMM, or CTC algorithm for end-to-end training
- test.py for training deep BLSTM with CTC is deprecated atm, don't use.
