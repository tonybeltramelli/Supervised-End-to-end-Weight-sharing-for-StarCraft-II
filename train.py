#!/usr/bin/env python
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys

from Dataset import *
from End2EndWeightSharingModel import *

import os.path

np.random.seed(1234)

argv = sys.argv[1:]

if len(argv) < 2:
    print("Error: not enough argument supplied:")
    print("train.py <name> <nb_epochs>")
    exit(0)
else:
    name = argv[0]
    epochs = int(argv[1])

dataset = Dataset()
dataset.load("dataset_{}".format(name))

model = End2EndWeightSharingModel()

image_input_shape = np.shape(dataset.input_observations)[1:]
actions_input_shape = np.shape(dataset.output_actions)[1:]
output_size = actions_input_shape[0]

if os.path.isfile("bin/agent_{}.h5".format(name)) and os.path.isfile("bin/agent_{}.json".format(name)):
    model.load("agent_{}".format(name))
    model.init_loaded_model()
else:
    model.init_model(image_input_shape=image_input_shape, actions_input_shape=actions_input_shape, output_size=output_size)

training_results = model.fit(dataset.input_observations, dataset.input_available_actions, dataset.output_actions, dataset.output_params, dataset.weights, epochs)
# print(training_results.history)
model.save("agent_{}".format(name))
