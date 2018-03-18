import os
import glob

replays = glob.glob('D:\StarCraft II\Replays\DefeatRoaches\DefeatRoaches_*.SC2Replay')
print(replays)
for r in replays:
    print('python transform_replay.py --replay "{}" --agent ObserverAgent.ObserverAgent'.format(r))

for r in replays:
    os.system('python transform_replay.py --replay "{}" --agent ObserverAgent.ObserverAgent'.format(r))