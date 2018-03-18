#!/usr/bin/env python

from pysc2.lib import features
import numpy as np
import uuid

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

class ObserverAgent():
    def __init__(self):
        self.states = []

    def step(self, time_step, action):
        observation = np.expand_dims(time_step.observation["screen"][_PLAYER_RELATIVE], axis=3)
        self.states.append(np.array([observation, time_step.observation["available_actions"], action.function, action.arguments]))

        if len(self.states) == 64:
            new_file_name = str(uuid.uuid1())
            np.save("dataset_{}/{}".format("roaches", new_file_name), np.array(self.states))
            self.states = []