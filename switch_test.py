"""
From: 
`Value-Decomposition Networks For Cooperative Multi-Agent Learning`
implementation from `https://github.com/koulanurag/ma-gym`
"""
import time, pickle

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

def test_env():    
    #env = Switch()
    env = gym.make('Switch2-v1') # v1 makes switch fully-observable
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    obs_n = env.reset()
    print(obs_n)

    #while not all(done_n):
    actions1 = [3, 0, 0, 1, 2, 4]
    actions2 = [4, 4, 4, 4, 4, 4]
    for actions in zip(actions1, actions2):
        env.render()
        print(f'actions = {actions}')
        obs_n, reward_n, done_n, info = env.step(actions)
        print(obs_n)
        ep_reward += sum(reward_n)
        time.sleep(1)
    env.close()

def learn_transition_table(env, n_steps=1000):
    """Explore environment to estimate transition probs & rewards
    (ignores agent 2)"""
    def convert_state(state):
        # ignore state of agent 2
        return state[1] + 7*state[0]
    
    def pick_action(state):
        state = convert_state(state)
        counts = action_count[state]
        return argmin(counts)
    
    def argmin(values):
        minimum = sorted([(val, idx) for idx, val in enumerate(values)])
        return minimum[0][1]

    n_states  = 7 * 3
    n_actions = 5 # 0: down, 1: left, 2: up, 3: right, 4: no-op
    transition_table, rewards = {}, {}
    for key in [(s, a) for s in range(n_states) for a in range(n_actions)]:
        transition_table[key] = [0.,] * n_states
    action_count = {}
    for state in range(n_states):
        action_count[state] = [0.,] * n_actions
    

    done = False
    state = env.reset()[0]
    for _ in range(n_steps):
        action = pick_action(state)
        (next_state, _), (reward, _), (done, _), _ = env.step([action, 4])
        transition_table[(convert_state(state), action)][convert_state(next_state)] += 1
        if (convert_state(state), action) not in rewards:
            rewards[(convert_state(state), action)] = reward
        action_count[convert_state(state)][action] += 1

        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    for key in transition_table:
        total = sum(transition_table[key])
        if total != 0:
            transition_table[key] = [p/total for p in transition_table[key]]
    return transition_table, rewards, action_count

def generate_dynamics(filename='game_dynamics.pkl', n_steps=1000000):
    env = gym.make('Switch2-v1') # v1 makes switch fully-observable
    transition_table, rewards, counts = learn_transition_table(env, n_steps=n_steps)
    for action in range(5):
        print(action, transition_table[(19, action)])
    print(counts)

    with open(filename, 'wb') as f:
        pickle.dump({'transition': transition_table,'rewards': rewards, 'counts': counts}, f)
    
    env.close()
    return transition_table, rewards, counts

def load_dynamics(filename='game_dynamics.pkl'):
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    return d['transition'], d['rewards'], d['counts']

if __name__ == '__main__':
    filename = 'game_dynamics3.pkl'
    #transition_table, rewards, counts = generate_dynamics(filename=filename, n_steps=10000000)
    transition_table, rewards, counts = load_dynamics(filename=filename)
    for state in range(21):
        n = 0
        for action in range(5):
            #print(state, action, transition_table[(state, action)])
            n += sum(transition_table[(state, action)])
        print(f'{state}: {n}')
    #for key in rewards:
    #    print(f'{key}: reward = {rewards[key]}')
    print('..')
    #transition_table[(13, 3)][13] = 1.0
    #transition_table[(13, 4)][13] = 1.0
    #with open(filename, 'wb') as f:
    #    pickle.dump({'transition': transition_table,'rewards': rewards, 'counts': counts}, f)
    
    #test_env()