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
        self.neighbors = []
        self.messages = []
        self.message_store = []
    
    def __repr__(self):
        return self.name
    
    def add_neighbor(self, factor):
        self.neighbors.append(factor)
    
    def send_message(self):
        # when node is requested to send message and list of incoming messages is empty,
        # this means that node is leaf node and most use init message
        if len(self.messages) == 0:
            message = np.ones(len(self.states)) if isinstance(self, Variable) else self.values
            for node in self.neighbors:
                node.messages.append(Message(self, node, message))
                print(f'Node {self.name} sends message to node {node.name}')
            return self.neighbors # return all neighbors
        else: # not a leaf
            in_values= [message.values for message in self.messages]
            in_nodes = [message.origin for message in self.messages]
            out_nodes = [node for node in self.neighbors if node not in in_nodes]
            print('')
            for node in out_nodes:
                print(f'Node {self.name} sends message to node {node.name}')
                if isinstance(self, Variable):
                    # from variable to factor: product of all in messages:
                    message = np.prod(in_values, axis=0)
                else: # factor node
                    message = self.values.T @ np.prod(in_values, axis=0)
                node.messages.append(Message(self, node, message))
            return out_nodes

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
    
    def compute_marginals(self):
        leafs = list(self.leafs())
        # 1. pick (arbitrary) root node
        #root = leafs.pop()
        root = self.factors[-1] # force uses of  as root - remove later
        leafs.remove(root)
        # 2. Propagate messages from leaves to root
        fringe = leafs[:]
        print(f'fringe = {fringe}')
        while len(fringe) > 0:
            node = fringe.pop(0)
            next_nodes = node.send_message()
            fringe = fringe + [node for node in next_nodes if node not in fringe]
            print(f'fringe = {fringe}')
        print('-----------------------------------------------------')
        # 3. Propagate messages from root to leaves
        fringe = [root]
        print(f'fringe = {fringe}')
        while len(fringe) > 0:
            node = fringe.pop(0)
            next_nodes = node.send_message()
            fringe = fringe + [node for node in next_nodes if node not in fringe]
            print(f'fringe = {fringe}')

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
    print(g.factors)
    g.compute_marginals()
    for node in g.vars + g.factors:
        print(node, node.messages)
