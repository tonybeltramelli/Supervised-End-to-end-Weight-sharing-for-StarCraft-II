#!/usr/bin/env python

from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

class ObserverAgent():
    def step(self, time_step, action):
        print("loop {}".format(time_step.observation["game_loop"]))
        print("{}".format(action))
        print("{}".format(action.__class__.__name__))
        #print("{}".format(time_step.observation["screen"][_PLAYER_RELATIVE]))
        print("{}".format(time_step.observation["available_actions"]))

        if time_step.observation["game_loop"] > 100:
            1/0

