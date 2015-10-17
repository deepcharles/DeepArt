#import numpy as np
import lasagne
from vgg19 import build_model
import cPickle as pickle

def load_weights(filename, output_layer):
	model = pickle.load(open(filename))
	#CLASSES = model['synset words']
	#MEAN_IMAGE = model['mean image']
	lasagne.layers.sett_all_param_values(output_layer, model['values'])
