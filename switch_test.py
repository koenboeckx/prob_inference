"""
From: 
`Value-Decomposition Networks For Cooperative Multi-Agent Learning`
implementation from `https://github.com/koulanurag/ma-gym`
"""
import time

import gym
import ma_gym

class Switch:
    "No longer needed because Switch has fully-observable version ('Switch2-v1')"
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
    
#env = Switch()
env = gym.make('Switch2-v1') # v1 makes switch fully-observable
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
print(obs_n)

#while not all(done_n):
actions2 = [3, 0, 0, 1, 2, 4]
actions1 = [4, 4, 4, 4, 4, 4]
for actions in zip(actions1, actions2):
    env.render()
    print(f'actions = {actions}')
    obs_n, reward_n, done_n, info = env.step(actions)
    print(obs_n)
    ep_reward += sum(reward_n)
    time.sleep(1)
env.close()

def learn_transition_table(env):
    # ignore agent 2
    pass
