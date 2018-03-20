#!/usr/bin/env python

from pysc2.lib import features, actions
import numpy as np

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

class ObserverAgent():
    def __init__(self):
        self.states = []

    def getStates(self):
        return self.states

    def step(self, time_step, action):
        observation = np.expand_dims(time_step.observation["screen"][_PLAYER_RELATIVE], axis=3)
        self.states.append(np.array([observation, time_step.observation["available_actions"], action.function, action.arguments]))

class NoNoOp(ObserverAgent):
    def step(self, time_step, action):
        if action.function != actions.FUNCTIONS.no_op.id:
            super(NoNoOp, self).step(time_step, action)