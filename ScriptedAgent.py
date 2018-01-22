__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'
# scripted agents taken from PySC2, credits to DeepMind
# https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py

import numpy
import uuid

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from Utils import *

GAME = "beacon"


class ScriptedAgent(base_agent.BaseAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)

        self.states = []

    def step(self, obs):
        super(ScriptedAgent, self).step(obs)

        observation = obs.observation["minimap"][5]
        observation = Utils.feature_array_to_img(observation, max_target_value=1.0)
        observation = Utils.resize_squared_img(observation, 84)
        #Utils.show(observation)

        if GAME == "beacon":
            if actions.FUNCTIONS.Move_screen.id in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]
                neutral_y, neutral_x = (player_relative == 3).nonzero()

                if not neutral_y.any():
                    action = actions.FUNCTIONS.no_op.id
                    params = []
                else:
                    target = [int(neutral_x.mean()), int(neutral_y.mean())]

                    action = actions.FUNCTIONS.Move_screen.id
                    params = [[0], target]
            else:
                action = actions.FUNCTIONS.select_army.id
                params = [[0]]
        elif GAME == "mineral":
            if actions.FUNCTIONS.Move_screen.id in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]
                neutral_y, neutral_x = (player_relative == 3).nonzero()
                player_y, player_x = (player_relative == 1).nonzero()
                if not neutral_y.any() or not player_y.any():
                    action = actions.FUNCTIONS.no_op.id
                    params = []
                else:
                    player = [int(player_x.mean()), int(player_y.mean())]
                    closest, min_dist = None, None
                    for p in zip(neutral_x, neutral_y):
                        dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                        if not min_dist or dist < min_dist:
                            closest, min_dist = p, dist
                    action = actions.FUNCTIONS.Move_screen.id
                    params = [[0], closest]
            else:
                action = actions.FUNCTIONS.select_army.id
                params = [[0]]

        self.states.append(np.array([observation, obs.observation["available_actions"], action, params]))

        if len(self.states) == 64:
            new_file_name = str(uuid.uuid1())

            np.save("dataset_{}/{}".format(GAME, new_file_name), np.array(self.states))

            self.states = []

        return actions.FunctionCall(action, params)
