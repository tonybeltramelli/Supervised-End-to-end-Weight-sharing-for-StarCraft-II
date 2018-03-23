__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'
# scripted agents taken from PySC2, credits to DeepMind
# https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py

import numpy as np
import uuid

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class ScriptedAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(ScriptedAgent, self).step(obs)

        # we expand dims because keras wants 4 dims for convolutions
        # observation = np.expand_dims(obs.observation["screen"][_SCREEN_PLAYER_RELATIVE], axis=3)
        screens = [obs.observation["screen"][_SCREEN_PLAYER_RELATIVE],
                   obs.observation["screen"][_SCREEN_SELECTED]]
        observation = np.stack(screens, axis=2)

        if self.game == "beacon":
            if actions.FUNCTIONS.Move_screen.id in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][_SCREEN_PLAYER_RELATIVE]
                neutral_y, neutral_x = (player_relative == 3).nonzero()

                if not neutral_y.any():
                    action = _NO_OP
                    params = []
                else:
                    target = [int(neutral_x.mean()), int(neutral_y.mean())]

                    action = _MOVE_SCREEN
                    params = [[0], target]
            else:
                action = _SELECT_ARMY
                params = [[0]]
        elif self.game == "mineral":
            if actions.FUNCTIONS.Move_screen.id in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][_SCREEN_PLAYER_RELATIVE]
                neutral_y, neutral_x = (player_relative == 3).nonzero()
                player_y, player_x = (player_relative == 1).nonzero()
                if not neutral_y.any() or not player_y.any():
                    action = _NO_OP
                    params = []
                else:
                    player = [int(player_x.mean()), int(player_y.mean())]
                    closest, min_dist = None, None
                    for p in zip(neutral_x, neutral_y):
                        dist = np.linalg.norm(np.array(player) - np.array(p))
                        if not min_dist or dist < min_dist:
                            closest, min_dist = p, dist
                    action = _MOVE_SCREEN
                    params = [[0], closest]
            else:
                action = _SELECT_ARMY
                params = [[0]]
        elif self.game == "minerals":
            if actions.FUNCTIONS.Move_screen.id in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][_SCREEN_PLAYER_RELATIVE]
                neutral_y, neutral_x = (player_relative == 3).nonzero()
                player_y, player_x = (player_relative == 1).nonzero()
                if not neutral_y.any() or not player_y.any():
                    action = _NO_OP
                    params = []
                else:
                    player = [int(player_x.mean()), int(player_y.mean())]
                    closest, min_dist = None, None
                    for p in zip(neutral_x, neutral_y):
                        dist = np.linalg.norm(np.array(player) - np.array(p))
                        if not min_dist or dist < min_dist:
                            closest, min_dist = p, dist
                    action = _MOVE_SCREEN
                    params = [[0], closest]
            else:
                action = _SELECT_ARMY
                params = [[0]]
        elif self.game == "roaches":
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][_SCREEN_PLAYER_RELATIVE]
                roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
                if not roach_y.any():
                    action = _NO_OP
                    params = [_NOT_QUEUED]
                else:
                    index = np.argmax(roach_y)
                    target = [roach_x[index], roach_y[index]]
                    action = _ATTACK_SCREEN
                    params = [_NOT_QUEUED, target]
            elif _SELECT_ARMY in obs.observation["available_actions"]:
                action = _SELECT_ARMY
                params = [_SELECT_ALL]
            else:
                action = _NO_OP
                params = [_NOT_QUEUED]

        self.states.append(np.array([observation, obs.observation["available_actions"], action, params]))

        if len(self.states) == 64:
            new_file_name = str(uuid.uuid1())

            np.save("dataset_{}/{}".format(self.game, new_file_name), np.array(self.states))

            self.states = []

        return actions.FunctionCall(action, params)


class AgentRoaches(ScriptedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.game = "roaches"
        self.states = []


class AgentBeacon(ScriptedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.game = "beacon"
        self.states = []


class AgentMineral(ScriptedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.game = "mineral"
        self.states = []


class AgentMinerals(ScriptedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.game = "minerals"
        self.states = []
