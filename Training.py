from QLearningAlgorithms import *


def train():
    # parameters
    alpha = 0.1
    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]
    actions = np.round(np.linspace(-1, 1, 15), decimals=1)
    for trial in range(5000):
        states = [0.5, 0.1]  # initial states
        stringStates = [str(x) for x in states]
        for t in np.arange(0, 20, 0.03):
            new_action = choose_action(stringStates, actions, qTables)
            # dynamic computed inside
            x, v = getNextStates(h1=new_action[0], h2=new_action[1], v=states[1], t=t, x_0=states[0], v_0=states[1])
            if np.abs(x) > 2 or np.abs(v) > 1: break  # the ball has fallen
            r = reward(x, v)
            states = np.round([x, v], decimals=1)  # state update
            states = checkstates(states, qTables)  # check if the states have a match in the discrete grid
            stringStates = [str(x) for x in states]
            qTables = distributed(qTables, r, stringStates[0], stringStates[1], actions, alpha)
    print(qTables[0])


train()
