"""
From: 
`Value-Decomposition Networks For Cooperative Multi-Agent Learning`
implementation from `https://github.com/koulanurag/ma-gym`
"""
import time, pickle
import numpy as np 

import gym
import ma_gym

from hmm_beta import transition_table, rewards, actions, states
action_prior = [1/len(actions),] * len(actions) # uniform prior

def compute_marginal_beta(state, beta):
    beta_s = 0
    for action in actions:
        beta_s += beta[(state, action)] * action_prior[action]
    return beta_s

def compute_betas(T=20):
    beta_sa = {}
    beta_s  = {}
    for t in range(T, 0, -1):
        beta_sa[t] = {}
        beta_s[t] = {}
        if t == T:
            for s in states:
                for a in actions:
                    beta_sa[t][(s, a)] = np.exp(rewards[(s, a)])
                beta_s[t][s] = compute_marginal_beta(s, beta_sa[t])
        else:
            for s in states:
                for a in actions:
                    beta_sa[t][(s, a)] = 0
                    for next_s in states:
                        beta_sa[t][(s, a)] += beta_s[t+1][next_s] * transition_table[(s,a)][next_s]
                    beta_sa[t][(s, a)] *= np.exp(rewards[(s, a)])
                beta_s[t][s] = compute_marginal_beta(s, beta_sa[t])
    return beta_sa, beta_s


if __name__ == '__main__':
    env = gym.make('Switch2-v0')
    done_n = [False for _ in range(env.n_agents)]
    print(env.n_agents)
    print(env.reset())
    print(transition_table)