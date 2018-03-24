#!/usr/bin/env python

from pysc2.lib import features, actions
import numpy as np

_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index


class ObserverAgent():
    def __init__(self):
        self.states = []

    def getStates(self):
        return self.states

    def step(self, time_step, action):
        # we expand dims because keras wants 4 dims for convolutions
        # observation = np.expand_dims(obs.observation["screen"][_SCREEN_PLAYER_RELATIVE], axis=3)
        screens = [time_step.observation["screen"][_SCREEN_PLAYER_RELATIVE],
                   time_step.observation["screen"][_SCREEN_SELECTED]]
        observation = np.stack(screens, axis=2)
        self.states.append(np.array([observation, time_step.observation["available_actions"], action.function, action.arguments]))


class NoNoOp(ObserverAgent):
    def step(self, time_step, action):
        if action.function != actions.FUNCTIONS.no_op.id:
            super(NoNoOp, self).step(time_step, action)