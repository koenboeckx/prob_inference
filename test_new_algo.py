import copy
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

import gym
import gym_gridworld

action_pos = {0: [0,0],    # noop
              1: [-1, 0],  # up
              2: [1,0],    # down
              3: [0,-1],   # left
              4: [0,1]}    # right
actions = {'noop': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4}

def remove_self(grid_map):
    gm = copy.deepcopy(grid_map)
    gm[gm == 4] = 0
    return gm

def pos_estimate(obs, grid_map):
    o, gm = remove_self(obs), remove_self(grid_map)
    corr =  signal.correlate2d(gm, o, mode='same')
    max_idx = np.unravel_index(np.argmax(corr), grid_map.shape)
    likelihood = np.exp(corr)/np.sum(np.exp(corr))
    log_likelihood = np.log(likelihood)
    return log_likelihood, max_idx

def compute_transition_model(grid_map):
    transition_model = {}
    m, n = grid_map.shape
    for i in range(m):
        for j in range(n):
            transition_model[pos_to_state((i,j))] = [0,] * len(actions)
            if grid_map[i, j] == 1: # wall, so doesn't matter
                for action in actions.values():
                    transition_model[pos_to_state((i,j))][action] = pos_to_state((i,j))
            else:
                for action in actions.values():
                    di, dj = actions[action]
                    if grid_map[i+di, i+dj] == 1: # meaning wall
                        transition_model[pos_to_state((i,j))][action]


    return transition_model

def state_to_pos(state):
    "state = pos linearly"
    m, n = env.start_grid_map.shape
    return state // m, state % n

def pos_to_state(pos):
    m, n = env.start_grid_map.shape
    return m*pos[0] + pos[1]

class GoalInference:
    def __init__(self, env):
        self.env = env
    
    def remove_self(self, grid_map):
        "Remove self (= 4) from map"
        gm = copy.deepcopy(grid_map)
        gm[gm == 4] = 0
        return gm

    def pos_estimate(self, obs, grid_map):
        "Estimate position in gridmap based on observation"
        o, gm = self.remove_self(obs), remove_self(grid_map)
        corr =  signal.correlate2d(gm, o, mode='same')
        max_idx = np.unravel_index(np.argmax(corr), grid_map.shape)
        likelihood = np.exp(corr)/np.sum(np.exp(corr))
        log_likelihood = np.log(likelihood)
        return log_likelihood, max_idx
    
    def compute_transition_model(self):
        transition_model = {}
        m, n = self.env.current_grid_map.shape
        for i in range(m):
            for j in range(n):
                transition_model[pos_to_state((i,j))] = [0,] * len(actions)
                if self.env.current_grid_map[i, j] == 1: # wall, so doesn't matter
                    for action in actions.values():
                        transition_model[pos_to_state((i,j))][action] = pos_to_state((i,j))
                else:
                    for action in actions.values():
                        di, dj = actions[action]
                        if self.env.current_grid_map[i+di, i+dj] == 1: # meaning wall
                            transition_model[pos_to_state((i,j))][action]


    return transition_model


if __name__ == '__main__':
    env = gym.make('gridworld-v0')
    compute_transition_model(env.current_grid_map)