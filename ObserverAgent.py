#!/usr/bin/env python

class ObserverAgent():
    def step(self, time_step, actions):
        print("{}".format(time_step.observation["game_loop"]))
