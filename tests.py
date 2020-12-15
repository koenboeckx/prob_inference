import unittest

import numpy as np
from sum_product import Variable, Factor, FactorGraph
from belief_networks import BNVariable, make_factor_graph
from gridworld import GridWorld
from hmm import HMM



class TestMessagePassing(unittest.TestCase):
    def test_simple_graph(self):
        g = FactorGraph()

        x1 = Variable('x1', 2) # init a variable with 2 states
        x2 = Variable('x2', 3) # init a variable with 3 states

        f12 = Factor('f12', np.array([
        [0.8,0.2],
        [0.2,0.8],
        [0.5,0.5]
        ])) # create a factor, node potential for p(x1 | x2)

        # connect the parents to their children
        g.add(f12)
        g.append(f12, x2) # order must be the same as dimensions in factor potential!
        g.append(f12, x1) # note: f12 potential's shape is (3,2), i.e. (x2,x1)

        g.compute_messages()
        self.assertAlmostEqual(g.compute_marginal(x1)[0], 0.5)
        self.assertAlmostEqual(g.compute_marginal(x2)[0], 0.33333333)

    def test_extended_graph(self):
            # definition of the (so far empty) factor graph
            g = FactorGraph()

            # declaration of the variables
            # arguments: name, # of possible values, value = ... if observed
            A = Variable('A', 2)
            B = Variable('B', 2)
            C = Variable('C', 2)
            D = Variable('D', 2)

            f1 = Factor('f1', np.array([
                [10, 1],
                [1, 10]
            ]))

            f2 = Factor('f2', np.array([
                [1, 10],
                [10, 1]
            ]))

            f3 = Factor('f3', np.array([
                [10, 1],
                [1, 10]
            ]))

            f4 = Factor('f4', np.array(
                [10, 1]
            ))

            g.add(f1)
            g.append(f1, A)
            g.append(f1, B)

            g.add(f2)
            g.append(f2, B)
            g.append(f2, C)

            g.add(f3)
            g.append(f3, B)
            g.append(f3, D)

            g.add(f4)
            g.append(f4, C)
            
            g.compute_messages()
            self.assertAlmostEqual(g.compute_marginal(A)[0], 0.22614576)
            self.assertAlmostEqual(g.compute_marginal(B)[0], 0.16528926)
            self.assertAlmostEqual(g.compute_marginal(C)[0], 0.90909091)
            self.assertAlmostEqual(g.compute_marginal(D)[0], 0.22614576)

    def test_extended_graph_with_evidence_1(self):
        # definition of the (so far empty) factor graph
        g = FactorGraph()

        # declaration of the variables
        # arguments: name, # of possible values, value = ... if observed
        A = Variable('A', 2)
        B = Variable('B', 2)
        C = Variable('C', 2, value=0)
        D = Variable('D', 2)

        f1 = Factor('f1', np.array([
            [10, 1],
            [1, 10]
        ]))

        f2 = Factor('f2', np.array([
            [1, 10],
            [10, 1]
        ]))

        f3 = Factor('f3', np.array([
            [10, 1],
            [1, 10]
        ]))

        f4 = Factor('f4', np.array(
            [10, 1]
        ))

        g.add(f1)
        g.append(f1, A)
        g.append(f1, B)

        g.add(f2)
        g.append(f2, B)
        g.append(f2, C)

        g.add(f3)
        g.append(f3, B)
        g.append(f3, D)

        g.add(f4)
        g.append(f4, C)
        
        g.compute_messages()
        self.assertAlmostEqual(g.compute_marginal(A)[0], 0.16528926)
        self.assertAlmostEqual(g.compute_marginal(B)[0], 0.09090909)
        self.assertAlmostEqual(g.compute_marginal(C)[0], 1.0)
        self.assertAlmostEqual(g.compute_marginal(D)[0], 0.16528926)

    def test_extended_graph_with_evidence_2(self):
        # definition of the (so far empty) factor graph
        g = FactorGraph()

        # declaration of the variables
        # arguments: name, # of possible values, value = ... if observed
        A = Variable('A', 2)
        B = Variable('B', 2, value=0)
        C = Variable('C', 2)
        D = Variable('D', 2)

        f1 = Factor('f1', np.array([
            [10, 1],
            [1, 10]
        ]))

        f2 = Factor('f2', np.array([
            [1, 10],
            [10, 1]
        ]))

        f3 = Factor('f3', np.array([
            [10, 1],
            [1, 10]
        ]))

        f4 = Factor('f4', np.array(
            [10, 1]
        ))

        g.add(f1)
        g.append(f1, A)
        g.append(f1, B)

        g.add(f2)
        g.append(f2, B)
        g.append(f2, C)

        g.add(f3)
        g.append(f3, B)
        g.append(f3, D)

        g.add(f4)
        g.append(f4, C)
        
        g.compute_messages()
        self.assertAlmostEqual(g.compute_marginal(A)[0], 0.90909091)
        self.assertAlmostEqual(g.compute_marginal(B)[0], 1.0)
        self.assertEqual(g.compute_marginal(C)[0], 0.5)
        self.assertAlmostEqual(g.compute_marginal(D)[0], 0.90909091)

    def test_three_factor_1(self):
        g = FactorGraph()

        # declaration of the variables
        # arguments: name, # of possible values, value = ... if observed
        A = Variable('A', 2)
        B = Variable('B', 2)
        C = Variable('C', 2)

        fA = Factor('fA', np.array(
            [0.3, 0.7]
        ))

        g.add(fA)
        g.append(fA, A)

        fB = Factor('fB', np.array(
            [0.4, 0.6]
        ))

        g.add(fB)
        g.append(fB, B)

        fABC = Factor('fABC', np.array(
            [[[0.1, 0.9],
            [0.4, 0.6]],

            [[0.5, 0.5],
            [0.9, 0.1]]
        ]))

        g.add(fABC)
        g.append(fABC, A)
        g.append(fABC, B)
        g.append(fABC, C)

        g.compute_messages()
        self.assertEqual(g.compute_marginal(C)[0],0.602)
    
    def test_gridworld(self):
        gw = GridWorld()
        self.assertEqual(gw.step(11, 2), (15, 0, True))
        self.assertEqual(gw.step(12, 2), (12, -1.0, False))

    def test_transform_belief_net(self):
        A = BNVariable('A', 2)
        A.add_propability_table(np.array([0.3, 0.7]))
        B = BNVariable('B', 2)
        B.add_propability_table(np.array([0.4, 0.6]))
        C = BNVariable('C', 2, parents=[A,B])
        C.add_propability_table(np.array(
            [[[0.1, 0.9],
            [0.4, 0.6]],

            [[0.5, 0.5],
            [0.9, 0.1]]])
        )
        factor_graph = make_factor_graph([A, B, C])
        factor_graph.compute_messages()
        for var, result in zip(factor_graph.vars, [0.3, 0.4, 0.602]):
            self.assertEqual(factor_graph.compute_marginal(var)[0], result)

class TestHMM(unittest.TestCase):
    def test_simple_hmm(self):
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
        result = hmm.filtered_posterior(index=2)
        for r, v in zip(result.values(), [0.0, 0.0, 1.0]):
            self.assertEqual(r, v)
        hmm.compute_beta()
        for r, v in zip(hmm.smoothed_posterior(index=2).values(), [0.0, 0.0, 1.0]):
            self.assertEqual(r, v)

if __name__ == '__main__':
    unittest.main()