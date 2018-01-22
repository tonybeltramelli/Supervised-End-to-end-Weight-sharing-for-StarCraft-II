#!/usr/bin/env python
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys
import numpy as np

from Utils import *
from Dataset import *
from End2EndWeightSharingModel import *

np.random.seed(1234)

argv = sys.argv[1:]

if len(argv) < 1:
    print "Error: not enough argument supplied:"
    print "train.py <name>"
    exit(0)
else:
    name = argv[0]

dataset = Dataset()
dataset.load("dataset_{}".format(name))

model = End2EndWeightSharingModel()

image_input_shape = np.shape(dataset.input_observations)[1:]
actions_input_shape = np.shape(dataset.output_actions)[1:]
output_size = actions_input_shape[0]

model.init_model(image_input_shape=image_input_shape, actions_input_shape=actions_input_shape, output_size=output_size)

model.fit(dataset.input_observations, dataset.input_available_actions, dataset.output_actions, dataset.output_params)
model.save("agent_{}".format(name))
