#!/usr/bin/env python
# coding: utf-8

# [github link](https://github.com/ilyakava/sumproduct)
# see also: https://ermongroup.github.io/cs228-notes/

# ## Interface
# ```python
# g = FactorGraph()
# 
# x1 = Variable('x1', 2) # init a variable with 2 states
# x2 = Variable('x2', 3) # init a variable with 3 states
# 
# f12 = Factor('f12', np.array([
#   [0.8,0.2],
#   [0.2,0.8],
#   [0.5,0.5]
# ])) # create a factor, node potential for p(x1 | x2)
# 
# # connect the parents to their children
# g.add(f12)
# g.append('f12', x2) # order must be the same as dimensions in factor potential!
# g.append('f12', x1) # note: f12 potential's shape is (3,2), i.e. (x2,x1)
# 
# g.compute_marginals() # -> [0.0]
# 
# g.nodes['x1'].marginal() # -> array([0.5, 0.5])
# ```

# TODO: in its current form, the method Node.send_message() can not accomodate
#       factors with three variables, what must be addressed

import numpy as np

def update_with_evidence(A, index, axis):
    """Helper function - sets all values of A to zero,
        except for index along axis"""
    cut_out = np.take(A, index, axis=axis)
    B = np.zeros_like(A)
    slc = [slice(None)] * len(B.shape)
    slc[axis] = slice(index, index+1)
    B[tuple(slc)] = cut_out.reshape(B[tuple(slc)].shape)
    return B

class Message:
    def __init__(self, origin, dest, values):
        self.origin = origin
        self.dest = dest
        self.values = values
    
    def __repr__(self):
        return f'Message from {self.origin} to {self.dest}: {self.values}'


class Node:
    def __init__(self, name):
        self.name = name
        self.neighbors = set()
        self.messages = {}
        self.incoming= set()  # keeps track of neighbor nodes from which messages have been received
        self.outgoing = set() # keeps track of neighbor nodes to which messages have been sent
    
    def __repr__(self):
        return self.name
    
    def add_neighbor(self, node):
        self.neighbors.add(node)
    
    def send_message(self):
        if len(self.neighbors) == 1: # leaf node
            message = np.ones(len(self.states)) if isinstance(self, Variable) else self.values
            for node in self.neighbors:
                node.receive(self, Message(self, node, message))
            return self.neighbors # return neighbor
        else: # not a leaf
            out_nodes = set()
            for node in self.neighbors.difference(self.outgoing):
                # only send if message is received from all other neighbors:
                if len(self.incoming.union([node])) == len(self.neighbors):
                    out_nodes.add(node)
                    self.outgoing.add(node)
                    other_nodes = list(self.incoming.difference([node]))
                    in_values = [self.messages[n].values for n in other_nodes]

                    if isinstance(self, Variable):
                        # from variable to factor: product of all in messages:
                        message = np.prod(in_values, axis=0)
                    elif isinstance(self, Factor):         
                        # from factor to variable: sum product of all in messages: 
                        idxs = [self.axes_vars[n] for n in other_nodes]
                        idxs.insert(0, self.axes_vars[node])
                        message = np.transpose(self.values, axes=idxs)
                        #message = self.values.copy()
                        for val in reversed(in_values): # reversed because multiplying goes from outside in
                            message = np.dot(message, val)

                    node.receive(self, Message(self, node, message))
            return out_nodes
    
    def receive(self, sender, message):
        self.incoming.add(sender)
        self.messages[sender] = message

class Variable(Node):
    def __init__(self, name, n_states, value=None):
        """`value` is the observed value of the Variable; 
            if not observerd: `value=None`"""
        super().__init__(name)
        self.states = list(range(n_states))
        if value is not None:
            assert value in self.states, f'Value {value} is not a valid value for variable {name}'
            self.value = value
        else:
            self.value = None


class Factor(Node):
    def __init__(self, name, array):
        super().__init__(name)
        assert isinstance(array, np.ndarray), f'´array has to be a numpy ndarray, not {type(array)}'
        self.values = array
        self.shape  = self.values.shape
        self.connected_vars = 0
        self.axes_vars = {} # keeps track of which variable is associated with 
                            # which axis (dimension) of the `self.values` array
        self.next_axis = 0
    
    def add_neighbor(self, node):
        self.neighbors.add(node)
        self.axes_vars[node] = self.next_axis
        self.next_axis += 1

class FactorGraph:
    def __init__(self):
        self.factors = []
        self.vars = []
    
    def add(self, factor):
        """Add a factor to the graph"""
        assert factor not in self.factors, f'Factor {factor.name} already defined'
        self.factors.append(factor)
    
    def append(self, factor, variable):
        """Add a variable to factor"""
        assert factor in self.factors, f'Factor {factor.name} not yet defined'
        # Check for correct size
        assert factor.values.shape[factor.connected_vars] == len(variable.states), \
             f'Variable {variable} has incorrect size (size = {len(variable.states)}, required {factor.values.shape[factor.connected_vars] })'
        factor.connected_vars += 1
        if variable not in self.vars:
            self.vars.append(variable)
        # check for evidence - adapt factor accordingly (Barber, p.89)
        if variable.value is not None: # variable is set (observed) to value
            # set all elements of factor potential that don't 
            # correspond to variable value to zero
            axis = factor.connected_vars - 1
            idx = variable.states.index(variable.value)
            factor.values = update_with_evidence(factor.values, idx, axis)

        factor.add_neighbor(variable)
        variable.add_neighbor(factor)

    def leafs(self):
        leafs = set()
        for node in self.vars + self.factors:
            # a leaf is a node that is missing one incoming message
            if len(node.neighbors) == 1:
                leafs.add(node)
        return leafs
    
    def compute_marginal(self, variable):
        prod = np.prod([m.values for m in variable.messages.values()], axis=0)
        return(prod/sum(prod))
        
    def compute_messages(self):
        """In a tree exact inference of all the marginals can be done by two passes of the
            sum-product algorithm
            TODO: use logarithm: multiplications can result in very small values 
                                → use logarithm and summation instead (Barber, p.90)
            """
        leafs = list(self.leafs())
        # 1. pick (arbitrary) root node
        root = leafs.pop()
        # 2. Propagate messages from leaves to root
        fringe = leafs[:]
        while len(fringe) > 0:
            node = fringe.pop(0)
            next_nodes = node.send_message()
            fringe = fringe + [node for node in next_nodes if node not in fringe]
        # 3. Propagate messages from root to leaves
        fringe = [root]
        while len(fringe) > 0:
            node = fringe.pop(0)
            next_nodes = node.send_message()
            fringe = fringe + [node for node in next_nodes if node not in fringe]

def test01():
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

    #g.compute_marginals() # -> [0.0]

    #g.nodes['x1'].marginal() # -> array([0.5, 0.5])
    return g

def test02():
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
    return g

def test03():
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
          [0.9, 0.1]]])
    )

    g.add(fABC)
    g.append(fABC, A)
    g.append(fABC, B)
    g.append(fABC, C)

    return g

if __name__ == '__main__':
    g = test03()
    g.compute_messages()
    for node in g.vars + g.factors:
        print(f'{node}: {node.messages}')
    print('')
    for var in g.vars:
        print(f'P({var}) = {g.compute_marginal(var)}')
