__author__ = 'charles'
import numpy as np
from lasagne.utils import floatX
IMAGE_W = 600
MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))