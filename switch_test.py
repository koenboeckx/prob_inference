"""
From: 
`Value-Decomposition Networks For Cooperative Multi-Agent Learning`
implementation from `https://github.com/koulanurag/ma-gym`
"""
import time

import gym
import ma_gym

class Switch:
    def __init__(self):
        self.env = gym.make('Switch2-v0')
        self.action_space = self.env.action_space
    
    @property
    def n_agents(self):
        return self.env.n_agents
    
    def reset(self):
        obs_n = self.env.reset()
        print(obs_n)
        return obs_n
    
    def step(self, obs_n):
        return self.env.step(obs_n)
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
    


env = Switch()
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
"""
while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
    time.sleep(1)
print(ep_reward)
"""
env.close()