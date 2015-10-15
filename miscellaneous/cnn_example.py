import os
import sys
sys.path.append(os.path.abspath(os.path.join('.', os.pardir)))

import theano
import lasagne
from lasagne.layers import dnn
import numpy as np

from data import StratifiedChunkLoader, ImageConventionConverter, ImageTransformerLoader

#theano.config.optimizer='fast_compile'

chunk_size = 512
batch_size = 32

data_loader = StratifiedChunkLoader(chunk_size=chunk_size)
data_loader = ImageTransformerLoader(data_loader)
data_loader = ImageConventionConverter(data_loader)

l_in = lasagne.layers.InputLayer(shape=(batch_size, 3, 256, 256))

# to add zeros on input's borders
l_pad1 = lasagne.layers.PadLayer(l_in, width = 1)

l_conv1 = dnn.Conv2DDNNLayer(
    l_pad1,
    num_filters=64,
    filter_size=(8, 8),
    stride = 4,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform()
)

l_pool1 = dnn.MaxPool2DDNNLayer(l_conv1, pool_size=(3, 3), stride = 2)

l_pad2 = lasagne.layers.PadLayer(l_conv1, width = 1)

l_conv2 = dnn.Conv2DDNNLayer(
    l_pad2,
    num_filters=64,
    filter_size=(3, 3),
    stride = 2,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform()
)

l_pool2 = dnn.MaxPool2DDNNLayer(l_conv2, pool_size=(3, 3), stride = 2)

l_dropout1 = lasagne.layers.DropoutLayer(l_pool2, p=0.5)

l_flatten1 = lasagne.layers.FlattenLayer(l_dropout1)

l_hidden1 = lasagne.layers.DenseLayer(
    l_flatten1,
    num_units=128,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform()
)

l_dropout2 = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

l_nonlinearity = lasagne.layers.NonlinearityLayer(l_dropout2, nonlinearity=lasagne.nonlinearities.sigmoid)

l_hidden2 = lasagne.layers.DenseLayer(
    l_nonlinearity,
    num_units=128,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform()
)

l_out = lasagne.layers.DenseLayer(
    l_hidden2,
    num_units=5,
    nonlinearity=lasagne.nonlinearities.softmax,
    W=lasagne.init.GlorotUniform()
)

# The method to add a regularizer has recently changed. I follow this post from
# Sander Dieleman: https://groups.google.com/forum/#!topic/lasagne-users/OkEswD6euk4
l2_param = 0.01
reg = l2_param * lasagne.regularization.regularize_network_params(
                                            l_out, lasagne.regularization.l2)

model = l_out

import time
from train import train

now = time.time()
num_epochs = 100
for epoch in train(model, data_loader, regularizer=reg,
                   chunk_size=chunk_size, batch_size=batch_size,
                   learning_rate=.02, num_epochs=num_epochs):
    print("Epoch {} of {} took {:.3f}s".format(epoch['number'] + 1, num_epochs, time.time() - now))
    now = time.time()
    print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
    print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
    print("  validation accuracy:\t\t{:.2f} %%".format(epoch['valid_accuracy'] * 100))
    

from train import test
from sklearn.metrics import confusion_matrix, accuracy_score
from kappa import kappa

y_pred, y_true = test(model, data_loader, chunk_size=chunk_size, batch_size=batch_size)
y_pred2 = np.argmax(np.array(y_pred), axis=1)

print confusion_matrix(y_true, y_pred2)
print accuracy_score(y_true, y_pred2)
print kappa(y_true, y_pred2)

from utils import save2s3

save2s3(l_out, "hoyt_exp.pickled")
