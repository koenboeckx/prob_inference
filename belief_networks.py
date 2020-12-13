import numpy as np 

from sum_product import Variable, Factor, FactorGraph

class BNVariable:
    def __init__(self, name, n_states, parents=None):
        self.name = name
        self.states = list(range(n_states))
        self.parents = parents
    
    def __repr__(self):
        return self.name
    
    def add_propability_table(self, table):
        self.table = table


def make_factor_graph(beliefnet):
    fg = FactorGraph()
    vars = {}
    parents = [node for node in beliefnet if node.parents is None]
    for node in parents:
        vars[node.name] = Variable(node.name, len(node.states))
        factor = Factor('f'+node.name, node.table)
        fg.add(factor)
        fg.append(factor, vars[node.name])
        
    
    to_do = [node for node in beliefnet if node not in parents]
    while len(to_do) > 0:
        next_nodes = [n for n  in to_do if all([p in parents for p in n.parents])] # only use nodes who have all parents already done
        for node in next_nodes:
            to_do.remove(node)
            #if not all([parent in parents for parent in node.parents]): # not all parents of node are already in factorgraph
            #    continue
            vars[node.name] = Variable(node.name, len(node.states))
            factor = Factor('f'+node.name, node.table)
            fg.add(factor)
            for parent in node.parents:
                fg.append(factor, vars[parent.name])
            fg.append(factor, vars[node.name])
            parents.append(node)

    return fg

if __name__ == '__main__':
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
    for node in factor_graph.vars + factor_graph.factors:
        print(f'{node}: {node.messages}')
    print('')
    for var in factor_graph.vars:
        print(f'P({var}) = {factor_graph.compute_marginal(var)}')