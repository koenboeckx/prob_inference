actions = ['up', 'right', 'down', 'left']

class GridWorld:
    def __init__(self, size=5, n_actions=len(actions)):
        self.P, self.reward = dict(), dict()
        for i in range(size):
            for j in range(size):
                self.P[(i,j)] = [None,]*n_actions
                self.reward[(i, j)] = [-1.,]*n_actions

if __name__ == '__main__':
    gw = GridWorld()
    print(gw.P)
    print(gw.reward)