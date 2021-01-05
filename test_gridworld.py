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