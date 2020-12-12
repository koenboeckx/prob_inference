import unittest

import numpy as np
from sum_product import Variable, Factor, FactorGraph


class Testing(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()