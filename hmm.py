""" Hidden Markov Model
Implements forward- and backward algorithm
(i.e. alpha and beta recursion)
"""
import numpy as np

class HiddenState:
    def __init__(self, values, parent, transition_table):
        self.values = values
        self.parent = parent
        self.table  = transition_table
    
    def __repr__(self):
        return f'Hidden state {self.values}'

class Observation:
    def __init__(self, values, hidden, observation_table):
        self.values = values
        self.hidden = hidden
        self.table  = observation_table
        self.observed = False
    
    def __repr__(self):
        return f'Obs state {self.values}'
    
    def set_observation(self, obs):
        assert obs in self.values, f'Illegal observation {obs} (valid: {self.values})'
        self.observed = obs

class HMM:
    def __init__(self, hidden_values, prior, transition_table,
                       obs_values, obs_table, n_steps):
        self.hidden_nodes = []
        for t in range(n_steps):
            if t == 0:
                node = HiddenState(hidden_values, None, prior)
            else:
                node = HiddenState(hidden_values, 
                                   self.hidden_nodes[t-1], transition_table)
            self.hidden_nodes.append((node))
        
        self.obs_nodes = []
        for t in range(n_steps):
            node = Observation(obs_values, self.hidden_nodes[t],
                               obs_table)
            self.obs_nodes.append(node)
    
    def __repr__(self):
        return f'Hidden Markov Model with: \n \thidden nodes: {self.hidden_nodes}\n\tobservation nodes: {self.obs_nodes}'
    
    def set_observed_values(self, observations):
        if len(observations) < len(self.obs_nodes): assert False, 'Not enough observations'
        if len(observations) > len(self.obs_nodes): assert False, 'Too much observations'
        for obs, node in zip(observations, self.obs_nodes):
            node.set_observation(obs)

    def compute_alpha(self):
        """Computes the alpha-coeffs used in the forward part
        of the forward-backward algorithm"""
        alphas = []
        for t, (hidden, obs) in enumerate(zip(self.hidden_nodes,
                                              self.obs_nodes)):
            alpha = {}
            for value in hidden.values:
                if t == 0:
                    alpha[value] = obs.table[value][obs.observed] * hidden.table[value]
                else:
                    prev_alpha = alphas[t-1]
                    corrector = obs.table[value][obs.observed]
                    predictor = sum([hidden.table[v][value] * prev_alpha[v] for v in hidden.values])
                    alpha[value] = corrector * predictor
            alphas.append(alpha)
        self.alphas = alphas
    
    def compute_beta(self):
        """Computes the beta-coeffs used in the backward part
        of the forward-backward algorithm"""
        betas = [None, ] * len(self.hidden_nodes)
        for t in range(len(self.hidden_nodes)-1, -1, -1):
            beta = {}
            if t == len(self.hidden_nodes)-1:
                for value in self.hidden_nodes[t].values:
                    beta[value] = 1.
                betas[t] =  beta
                continue
            hidden = self.hidden_nodes[t+1]
            obs = self.obs_nodes[t+1] # ! observation of node to the right
            prev_beta = betas[t+1]
            for value in hidden.values:
                beta[value] = sum([obs.table[v][obs.observed] * hidden.table[value][v] * prev_beta[v]
                                  for v in hidden.values])
            betas[t] =  beta
        self.betas = betas
    
    def compute_mus(self):
        """Computes the mu-coeffs used in the Viterbi algorithm"""
        mus = [None, ] * len(self.hidden_nodes)
        for t in range(len(self.hidden_nodes)-1, -1, -1):
            mu = {}
            if t == len(self.hidden_nodes)-1:
                for value in self.hidden_nodes[t].values:
                    mu[value] = 1.
            else:
                hidden = self.hidden_nodes[t+1]
                obs = self.obs_nodes[t+1] # ! observation of node to the right
                prev_mu = mus[t+1]
                for value in hidden.values:
                    mu[value] = max([obs.table[v][obs.observed] * hidden.table[value][v] * prev_mu[v]
                                    for v in hidden.values])
            mus[t] =  mu
        self.mus = mus
    
    def filtered_posterior(self, index):
        "Computes p(h[index] | previous observation)"
        assert hasattr(self, 'alphas'), 'First compute alphas'
        alpha = self.alphas[index].copy()
        normalizer = sum(alpha.values())
        for value in alpha:
            alpha[value] /= normalizer
        return alpha
    
    def smoothed_posterior(self, index):
        "Computes p(h[index] | all observation)"
        assert hasattr(self, 'alphas'), 'First compute alphas (`compute_alpha`)'
        assert hasattr(self, 'betas'), 'First compute betas (`compute_beta`)'
        alpha = self.alphas[index]
        beta  = self.betas[index]
        posterior = {}
        for value in alpha:
            posterior[value] = alpha[value]*beta[value]
        normalizer = sum(posterior.values())
        for value in posterior:
            posterior[value] /= normalizer
        return posterior
    
    def viterbi(self):
        hs = []
        for t in range(len(self.hidden_nodes)):
            h_star, h_value = None, -float('inf')
            obs = self.obs_nodes[t]
            hidden = self.hidden_nodes[t]
            for h in self.hidden_nodes[t].values:
                if t == 0:
                    val = obs.table[h][obs.observed] * hidden.table[h] * self.mus[t][h]
                else:
                    val = obs.table[h][obs.observed] * hidden.table[hs[t-1]][h] * self.mus[t][h]
                if val > h_value:
                    h_star, h_value = h, val
            hs.append(h_star)
        return hs

# ----------- tests -----------------------------
def test01():
    hidden_values = ['S', 'R']
    prior = {'S': 0.2, 'R': 0.8}
    transition_table = {
        'S': {'S': 0.8, 'R': 0.2},
        'R': {'S': 0.4, 'R': 0.6}
    }
    obs_values = ['H', 'G']
    obs_table = {
        'S': {'H': 0.8, 'G': 0.2},
        'R': {'H': 0.4, 'G': 0.6}
    }
    hmm = HMM(hidden_values, prior, transition_table, obs_values, obs_table,
              n_steps=5)
    hmm.set_observed_values(['H', 'H', 'G', 'G', 'H'])
    hmm.compute_alpha()
    print(hmm.filtered_posterior(index=2))
    hmm.compute_beta()
    print(hmm.smoothed_posterior(index=3))
    hmm.compute_mus()
    print(hmm.mus)
    print(hmm.viterbi())

def test02():
    "https://people.csail.mit.edu/rameshvs/content/hmms.pdf"
    hidden_values = ['Area 1', 'Area 2', 'Area 3']

    prior = {'Area 1': 1/3, 'Area 2': 1/3, 'Area 3': 1/3}

    transition_table = {
        'Area 1': {'Area 1': 0.25, 'Area 2': 0.75, 'Area 3': 0.},
        'Area 2': {'Area 1': 0.  , 'Area 2': 0.25, 'Area 3': 0.75},
        'Area 3': {'Area 1': 0.  , 'Area 2': 0.  , 'Area 3': 1.},
    }

    obs_values = ['hot', 'cold']
    obs_table = {
        'Area 1': {'hot': 1.0, 'cold': 0.0},
        'Area 2': {'hot': 0.0, 'cold': 1.0},
        'Area 3': {'hot': 1.0, 'cold': 0.0},
    }

    hmm = HMM(hidden_values, prior, transition_table, obs_values, obs_table,
              n_steps=3)
    hmm.set_observed_values(['hot', 'cold', 'hot'])

    hmm.compute_alpha()
    print(hmm.filtered_posterior(index=2))
    hmm.compute_beta()
    print(hmm.smoothed_posterior(index=2))
    print('alpha: ', hmm.alphas)
    print('beta: ', hmm.betas)

if __name__ == '__main__':
    test01()