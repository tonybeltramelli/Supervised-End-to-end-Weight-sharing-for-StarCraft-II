# Supervised End-to-end Weight-sharing for StarCraft II

<img src="StarCraftAIWorkshop.png?raw=true"/>

## What
This is the source code for the project I built during the StarCraft 2 AI Workshop organised by [Niels Justesen](https://njustesen.com/) and [Prof. Sebastian Risi](http://sebastianrisi.com/) at the [IT University of Copenhagen](https://www.itu.dk/). Many thanks to them and the sponsors for making this workshop possible *(header image credits to Niels Justesen)*. Turns out, I was fortunate enough to win the first price, yeah!!!!! :smiley: :thumbsup:

## Disclaimer
Sorry in advance, the code is far from clean and not very well-structured. Please keep in mind that the goal of the workshop was to implement something in less than __26 hours__ from 1pm on Saturday January 20th to 3pm on Sunday January 21st.

## Dependencies
* StarCraft 2 for AI research: https://starcraft2.com/en-us/#free-to-play
* DeepMind PySC2 (1.2): https://github.com/deepmind/pysc2
* Mini-game maps: https://github.com/deepmind/pysc2/tree/master/pysc2/maps/mini_games
* Python (2.7.13): https://www.python.org/downloads/
* TensorFlow (1.4.0): https://github.com/tensorflow/tensorflow
* Keras (2.0.8): https://github.com/keras-team/keras

In parentheses is the version of these packages I used during this workshop but you should be able to run my code using other releases with minor changes.
You can read [Niel's great post](https://njustesen.com/2018/01/16/getting-started-with-the-starcraft-2-learning-environment/) for a step by step walkthrough on setting up your environment.

## Goal

I had absolutely zero knowledge about StarCraft when I started this workshop so I could not input a ton of prior in my agent. To make things a little easier, I used the mini games as a test bed instead of the complex maps of the full game. Reinforcement Learning for StarCraft turned out to be more complex than anticipated so my final solution is leveraging Supervised Learning to train the agent. The training dataset is gathered by recording moves of scripted agents provided in the [DeepMind PySC2 package](https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py).

My model has two inputs and two outputs. It takes an image as input (minimap player_relative) together with a one-hot encoded vector representing all the available actions for a given game state. The model is then designed to predict policy: both the next action to take, and the (x, y) coordinates of the screen where to click. The model is trained end-to-end to perform both classification (next action) and regression (screen coordinates). The weights of the convolutional layers learning visual features are shared.

<img src="model_architecture.png?raw=true"/>

## Results

##### MoveToBeacon mini-game
![](https://raw.githubusercontent.com/tonybeltramelli/Supervised-End-to-end-Weight-sharing-for-StarCraft-II/master/beacon_agent_demo.gif)
* Gif demo: [beacon_agent_demo.gif](beacon_agent_demo.gif)
* Limitations: the agent performs really well in this game but sometimes cannot move to rightmost positions on the screen. This is probably because the (x, y) coordinates predicted by the model are screen ratios from [0.0, 1.0] which are then transformed to screen coordinates. My guess is that somewhere along the way, the value is rounded to the nearest integer below its current value.

##### CollectMineralShards mini-game
![](https://raw.githubusercontent.com/tonybeltramelli/Supervised-End-to-end-Weight-sharing-for-StarCraft-II/master/mineral_agent_demo.gif)
* Gif demo: [mineral_agent_demo.gif](mineral_agent_demo.gif)
* Limitations: the agent is performing poorly in this game. Both characters are attracted to each other instead of actively searching for minerals. My guess is that the model actually predict the mean of the mineral cluster position instead of the coordinates of the nearest mineral. This could explain why the characters are always sprinting to a position near the board's center.

## Usage

Run my pre-trained models in *./bin*:
```sh
# play the game with trained agent
# load the correct pre-trained model by changing self.model.load("agent_beacon") in TrainedAgent.py (yeah I know...)
python -m pysc2.bin.agent --map MoveToBeacon --agent TrainedAgent.TrainedAgent
python -m pysc2.bin.agent --map CollectMineralShards --agent TrainedAgent.TrainedAgent
```

I have made my datasets available *(dataset_beacon.zip, dataset_mineral.zip)* but you can gather your own data as follows:
```sh
mkdir dataset_beacon
mkdir dataset_mineral
# generate training data using scripted agents
# change variable GAME = "mineral|beacon" in ScriptedAgent.py (no argument sorry...)
python -m pysc2.bin.agent --map MoveToBeacon --agent ScriptedAgent.ScriptedAgent --max_agent_steps 10000
python -m pysc2.bin.agent --map CollectMineralShards --agent ScriptedAgent.ScriptedAgent --max_agent_steps 10000
```

Train your own model:
```sh
# train the model
python train.py <beacon|mineral|roaches> <epochs>
```

## Citation
If this work is useful to your research, please cite it as follows.

```
@online{beltramelli2017starcraft,
  title={Supervised End-to-end Weight-sharing for StarCraft II},
  author={Beltramelli, Tony},
  url={https://github.com/tonybeltramelli/Supervised-End-to-end-Weight-sharing-for-StarCraft-II},
  year={2017}
}
```

## License

This project and the associated media are distributed under the
[Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/), the source code is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).
