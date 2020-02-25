from QLearningAlgorithms import *


def train():
    # parameters
    alpha = 0.1
    # generate Q-Table
    qTables = 0
    actions_indexes = range(15)
    actions = np.linspace(-1, 1, 15)
    for trial in range(5000):
        states = [0.5, 0.1]  # initial states
        for t in np.arange(0, 20, 0.03):
            new_action = choose_action(states, actions, qTables)
            # dynamic computed inside
            x, v = getNextStates(h1=new_action[0], h2=new_action[1], v=states[1], t=t, x_0=states[0], v_0=states[1])
            r = reward(x, v)
            qTables = distributed(qTables, r, x, v, actions_indexes, alpha)
            states = [x, v]
    print(qTables)
