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

TERMINAL_VAL = 10
actions = {0:'up', 1:'right', 2:'down', 3:'left'}

class GridWorld:
    def __init__(self, size=4, n_actions=len(actions)):
        self.P, self.R = dict(), dict()
        self.n_states = size**2
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
                elif a == 1:                    # left
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

if __name__ == '__main__':
    gw = GridWorld()
    print(gw.step(12, 2))