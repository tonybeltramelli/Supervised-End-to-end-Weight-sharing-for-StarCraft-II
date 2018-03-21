__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from End2EndWeightSharingModel import *

np.random.seed(1234)
_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index


class TrainedAgent(base_agent.BaseAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)

        self.model = End2EndWeightSharingModel()
        # self.model.load("agent_beacon")
        #self.model.load("agent_mineral")
        # self.model.load("agent_minerals")
        self.model.load("agent_roaches")

    def step(self, obs):
        super(TrainedAgent, self).step(obs)

        # we expand dims because keras wants 4 dims for convolutions
        # observation = np.expand_dims(obs.observation["screen"][_SCREEN_PLAYER_RELATIVE], axis=3)
        screens = [obs.observation["screen"][_SCREEN_PLAYER_RELATIVE],
                   obs.observation["screen"][_SCREEN_SELECTED]]
        observation = np.stack(screens, axis=2)

        output_size = len(actions.FUNCTIONS)

        available_actions = np.zeros(output_size)
        for action_index in obs.observation["available_actions"]:
            available_actions[action_index] = 1.0

        input_batch = [np.array([observation]), np.array([available_actions])]
        action, position = self.model.predict(input_batch)
        screen_size = np.shape(obs.observation["screen"])[1]
        x = int(screen_size * position[0])
        y = int(screen_size * position[1])

        if action in obs.observation["available_actions"]:
            # print("action is available: ", action, x, y)
            if action == 7:
                print("select army")
        else:
            # if action not in obs.observation["available_actions"]:
            print("action", action, "is not available")
            action = np.random.choice(obs.observation["available_actions"])
            print("action", action, "was produced")

        if action == actions.FUNCTIONS.no_op.id:
            params = []
        elif action == actions.FUNCTIONS.Move_screen.id:
            params = [[0], [x, y]]
        elif action == actions.FUNCTIONS.select_army.id:
            params = [[0]]
        elif action == actions.FUNCTIONS.Attack_screen.id:
            params = [[0], [x, y]]
        else:
            params = [[np.random.randint(0, size) for size in arg.sizes] for arg in
                      self.action_spec.functions[action].args]

        return actions.FunctionCall(action, params)
