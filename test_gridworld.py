import copy
from matplotlib import pyplot as plt

import gym
import gym_gridworld
env = gym.make('gridworld-v0')
env.verbose = False

print('start  = ', env.get_start_state())
print('target = ', env.get_target_state())
obs = env.reset()


plt.imshow(obs)
plt.show()
#obs, reward, done, info = env.step(env.action_space.sample())

print(env.current_grid_map)
print(obs)
print(obs.shape)

plt.imshow(env.current_grid_map)
plt.show()

def remove_self(grid_map):
    gm = copy.deepcopy(grid_map)
    gm[gm == 4] = 0
    return gm

def pos_estimate(obs, grid_map):
    o, gm = remove_self(obs), remove_self(grid_map)

