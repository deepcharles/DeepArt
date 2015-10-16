#import numpy as np
import lasagne

from vgg19 import build_model

import cPickle as pickle
model = pickle.load(open('data/vgg19.pkl'))
CLASSES = model['synset words']
MEAN_IMAGE = model['mean image']
lasagne.layers.sett_all_param_values(output_layer, model['values'])