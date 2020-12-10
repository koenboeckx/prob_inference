#!/usr/bin/env python
# coding: utf-8

# [github link](https://github.com/ilyakava/sumproduct)

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

import numpy as np

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
                    other_nodes = self.incoming.difference([node])
                    in_values = [self.messages[n].values for n in other_nodes]
                    
                    if isinstance(self, Variable):
                        # from variable to factor: product of all in messages:
                        message = np.prod(in_values, axis=0)
                    else: # factor node
                        message = self.values.T @ np.prod(in_values, axis=0)
                    node.receive(self, Message(self, node, message))
            return out_nodes
    
    def receive(self, sender, message):
        self.incoming.add(sender)
        self.messages[sender] = message

class Variable(Node):
    def __init__(self, name, n_states):
        super().__init__(name)
        self.states = list(range(n_states))    


class Factor(Node):
    def __init__(self, name, array):
        super().__init__(name)
        assert isinstance(array, np.ndarray), f'´array has to be a numpy ndarray, not {type(array)}'
        self.values = array
        self.shape  = self.values.shape

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
        # TODO: check for correct size
        if variable not in self.vars:
            self.vars.append(variable)
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
        leafs = list(self.leafs())
        # 1. pick (arbitrary) root node
        #root = leafs.pop()
        root = self.factors[-1] # force uses of  as root - remove later
        leafs.remove(root)
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
    ## second test
    g = FactorGraph()

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

if __name__ == '__main__':
    g = test02()
    print(g.vars)
    g.compute_messages()
    for node in g.vars + g.factors:
        print(node, node.messages)
    for var in g.vars:
        print(f'{var}: {g.compute_marginal(var)}')
