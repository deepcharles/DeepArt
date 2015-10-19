import os
import sys
sys.path.append(os.path.abspath(os.path.join('.', os.pardir)))

import theano
import lasagne
from lasagne.layers import dnn
import numpy as np 

#theano.config.optimize='fast_compile'

#########################
#########################
### CNN ARCHITECTURE ####
#########################
#########################

""" VGG_ILSVRG_19_layers"""

data = lasagne.layers.InputLayer(shape=(10, 3, 224, 224))

###################
## Layer block 1 ##
###################

pad1_1 = lasagne.layers.PadLayer(data, width=1)

conv1_1 = dnn.Conv2DDNNLayer(
	pad1_1,
	num_filters=64,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad1_2 = lasagne.layers.PadLayer(conv1_1, width=1)

conv1_2 = dnn.Conv2DDNNLayer(
	pad1_2,
	num_filters=64,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pool1 = dnn.MaxPool2DDNNLayer(conv1_2, pool_size=2, stride=2)

###################
## Layer block 2 ##
###################

pad2_1 = lasagne.layers.PadLayer(pool1, width=1)

conv2_1 = dnn.Conv2DDNNLayer(
	pad2_1,
	num_filters=128,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad2_2 = lasagne.layers.PadLayer(conv2_1, width=1)

conv2_2 = dnn.Conv2DDNNLayer(
	pad2_2,
	num_filters=128,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pool2 = dnn.MaxPool2DDNNLayer(conv2_2, pool_size=2, stride=2)

###################
## Layer block 3 ##
###################

pad3_1 = lasagne.layers.PadLayer(pool2, width=1)

conv3_1 = dnn.Conv2DDNNLayer(
	pad3_1,
	num_filters=256,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad3_2 = lasagne.layers.PadLayer(conv3_1, width=1)

conv3_2 = dnn.Conv2DDNNLayer(
	pad3_2,
	num_filters=256,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad3_3 = lasagne.layers.PadLayer(conv3_2, width=1)

conv3_3 = dnn.Conv2DDNNLayer(
	pad3_3,
	num_filters=256,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad3_4 = lasagne.layers.PadLayer(conv3_3, width=1)

conv3_4 = dnn.Conv2DDNNLayer(
	pad3_4,
	num_filters=256,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pool3 = dnn.MaxPool2DDNNLayer(conv2_4, pool_size=2, stride=2)

###################
## Layer block 4 ##
###################

pad4_1 = lasagne.layers.PadLayer(pool3, width=1)

conv4_1 = dnn.Conv2DDNNLayer(
	pad4_1,
	num_filters=512,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad4_2 = lasagne.layers.PadLayer(conv4_1, width=1)

conv4_2 = dnn.Conv2DDNNLayer(
	pad4_2,
	num_filters=512,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad4_3 = lasagne.layers.PadLayer(conv4_2, width=1)

conv4_3 = dnn.Conv2DDNNLayer(
	pad4_3,
	num_filters=512,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad4_4 = lasagne.layers.PadLayer(conv4_3, width=1)

conv4_4 = dnn.Conv2DDNNLayer(
	pad4_4,
	num_filters=512,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pool4 = dnn.MaxPool2DDNNLayer(conv4_4, pool_size=2, stride=2)

###################
## Layer block 5 ##
###################

pad5_1 = lasagne.layers.PadLayer(pool4, width=1)

conv5_1 = dnn.Conv2DDNNLayer(
	pad5_1,
	num_filters=512,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad5_2 = lasagne.layers.PadLayer(conv5_1, width=1)

conv5_2 = dnn.Conv2DDNNLayer(
	pad5_2,
	num_filters=512,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad5_3 = lasagne.layers.PadLayer(conv5_2, width=1)

conv5_3 = dnn.Conv2DDNNLayer(
	pad5_3,
	num_filters=512,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pad5_4 = lasagne.layers.PadLayer(conv5_3, width=1)

conv5_4 = dnn.Conv2DDNNLayer(
	pad5_4,
	num_filters=512,
	filter_size=3,
	nonlinearity=lasagne.nonlinearities.rectify,
	W=lasagne.init.GlorotUniform()
)

pool5 = dnn.MaxPool2DDNNLayer(conv5_4, pool_size=2, stride=2)

###################
## Layer block 6 ##
###################

fc6 = lasagne.layers.DenseLayer(
    pool5,
    num_units=4096,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform()
)

drop6 = lasagne.layers.DropoutLayer(fc6, p=0.5)

###################
## Layer block 7 ##
###################

fc7 = lasagne.layers.DenseLayer(
    drop6,
    num_units=4096,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform()
)

drop7 = lasagne.layers.DropoutLayer(fc7, p=0.5)

###################
## Layer block 8 ##
###################

fc8 = lasagne.layers.DenseLayer(
    drop7,
    num_units=1000,
    nonlinearity=lasagne.nonlinearities.softmax,
    W=lasagne.init.GlorotUniform()
)

############################
############################
### END OF ARCHITECTURE ####
############################
############################
