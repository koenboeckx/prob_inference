"""
Sutton - Example 4.1 - 4×4 gridworld
-----------------
| T | 1 | 2 | 3 |
-----------------
| 4 | 5 | 6 | 7 |
-----------------
| 8 | 9 |10 |11 |
-----------------
|12 |13 |14 | T |
-----------------
The nonterminal states are S = {1, 2, . . . , 14}. There are four actions
possible in each state, A = {up, down, right, left}, which deterministically 
cause the corresponding state transitions, except that actions that
would take the agent off the grid in fact leave the state unchanged. Thus,
for instance, p(6, −1|5, right) = 1, p(7, −1|7, right) = 1, and 
p(10, r|5, right) = 0 for all r ∈ R. This is an undiscounted, episodic task. 
The reward is −1 on all transitions until the terminal state is reached.
The terminal state is shaded in the figure (although it is shown in two places,
it is formally one state). The expected reward function is thus r(s, a, s') = −1 
for all states s, s' and actions a.

"""
import numpy as np
from belief_networks import BNVariable

TERMINAL_VAL = 0
actions = {0:'up', 1:'right', 2:'down', 3:'left'}

class GridWorld:
    def __init__(self, size=4, n_actions=len(actions)):
        self.P, self.R = dict(), dict()
        self.n_states  = size**2
        self.n_actions = n_actions
        self.terminal_states = [0, 15]
        for s in range(self.n_states):
            for a in range(n_actions):
                self.P[(s, a)] = [0,]*self.n_states
                self.R[(s, a)] = -1.
                if   a == 0:                    # up 
                    if s >= 4: self.P[(s, a)][s-4] = 1
                    else: self.P[(s, a)][s] = 1
                elif a == 1:                    # right
                    if s not in [3,7,11,15]:   
                        self.P[(s, a)][s+1] = 1
                    else: self.P[(s, a)][s] = 1
                elif a == 2:                    # down
                    if s <= 11: self.P[(s, a)][s+4] = 1
                    else: self.P[(s, a)][s] = 1
                elif a == 3:                    # left
                    if s not in [0,4,8,12]:    
                        self.P[(s, a)][s-1] = 1
                    else: self.P[(s, a)][s] = 1
        #self.R[( 1, 3)] = TERMINAL_VAL # (s= 1, 'left')
        #self.R[( 4, 0)] = TERMINAL_VAL # (s= 4, 'up')
        #self.R[(11, 2)] = TERMINAL_VAL # (s=11, 'down')
        #self.R[(14, 1)] = TERMINAL_VAL # (s=14, 'right')
        for s in self.terminal_states:
            for a in range(n_actions):
                self.R[( s, a)] = TERMINAL_VAL
    
    def step(self, s, a):
        p = self.P[(s, a)]
        next_s = np.random.choice(range(self.n_states), p=p)
        reward = self.R[(next_s, a)]
        return next_s, reward, next_s in self.terminal_states

def make_beliefnet(gridworld, n_steps=10):
    """Creates a (static) belief network based on environment dynamics
    Refs:
        [1]: Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review
             2018, Sergey Levine (UC Berkeley) - arxiv.org/abs/1805.00909v3
    """
    # create the (fixed) propability tables for the three kind of variables
    # 1. state var -> transition table
    tansition_table = np.zeros((gridworld.n_states, gridworld.n_actions, gridworld.n_states))
    for s in range(gridworld.n_states):
        for a in range(gridworld.n_actions):
            tansition_table[s, a, :] = gridworld.P[(s, a)]
    # 2. action var -> uniform prior
    action_table = np.ones(gridworld.n_actions)/gridworld.n_actions
    # 3. optimality var -> exponential over rewards (see ref. 1)
    optimality_table = np.zeros((gridworld.n_states, gridworld.n_actions, 2))
    for s in range(gridworld.n_states):
        for a in range(gridworld.n_actions):
            optimality_table[s, a, :] = np.array([    np.exp(gridworld.R[( s, a)]),
                                                  1 - np.exp(gridworld.R[( s, a)])])

    states, actions, optimals = [], [], []
    for t in range(n_steps):
        state = BNVariable('s' + str(t), gridworld.n_states,
                parents=None if t == 0 else [states[t-1], actions[t-1]])
        state.add_propability_table(tansition_table)
        action = BNVariable('a' + str(t), gridworld.n_actions)
        action.add_propability_table(action_table)
        optimal = BNVariable('o' + str(t), 2, parents=[state, action])
        optimal.add_propability_table(optimality_table)
        states.append(state)
        actions.append(action)
        optimals.append(optimal)
    return states + actions + optimals


if __name__ == '__main__':
    gw = GridWorld()
    beliefnet = make_beliefnet(gw, n_steps=3)
    for var in beliefnet:
        print(np.sum(var.table, axis=-1))
    print('---------------------------------------------------')
    print(beliefnet[0].table[0,:,:].shape)
    print(beliefnet[0].table[0,:,:])
    from belief_networks import make_factor_graph
    factor_graph = make_factor_graph(beliefnet)
    factor_graph.compute_messages()
    
    for var in factor_graph.vars:
        print(f'P({var}): {factor_graph.compute_marginal(var)}')
    