from QLearningAlgorithms import *
from Utility import *
import pandas as pd


def trainDistributed():
    # parameters
    alpha = 0.9

    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]
    actions = np.round(np.linspace(-1, 1, 15), decimals=1)

    for trial in range(5000):
        states = (0.5, 0.1)  # initial states
        for t in np.arange(0, 20, 0.03):
            new_action = choose_action(states, actions, qTables)

            # dynamic computed inside
            x, v = getNextStates(h1=new_action[0], h2=new_action[1], v=states[1], t=t, x_0=states[0], v_0=states[1])
            if np.abs(x) > 2 or np.abs(v) > 1: break  # the ball has fallen

            r = reward(x, v)
            states = (np.round(x, decimals=1), np.round(v, decimals=1))
            states = checkstates(states, qTables)  # check if the states have a match in the discrete grid
            qTables = distributed(qTables, r, states, actions, alpha)
    print(qTables[0])
    countNot0(qTables)


def trainDecentralized():
    # parameters
    alpha = 0.9
    gamma = 0.9

    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]
    actions = np.round(np.linspace(-1, 1, 15), decimals=1)

    for trial in range(5000):
        states = (0.5, 0.1)  # initial states
        for t in np.arange(0, 20, 0.03):
            new_action = choose_action(states, actions, qTables)

            # dynamic computed inside
            x, v = getNextStates(h1=new_action[0], h2=new_action[1], v=states[1], t=t, x_0=states[0], v_0=states[1])
            if np.abs(x) > 2 or np.abs(v) > 1: break  # the ball has fallen

            r = reward(x, v)
            new_states = (np.round(x, decimals=1), np.round(v, decimals=1))
            new_states = checkstates(new_states, qTables)  # check if the states have a match in the discrete grid
            qTables = decentralized(qTables, states, actions, alpha, r, gamma, new_states)
            states = new_states
    print(qTables[0])
    countNot0(qTables)


def trainHysteretic():
    # parameters
    alpha = 0.9
    beta = 0.1
    gamma = 0.9

    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]
    actions = np.round(np.linspace(-1, 1, 15), decimals=1)

    for trial in range(5000):
        states = (0.5, 0.1)  # initial states
        for t in np.arange(0, 20, 0.03):
            new_action = choose_action(states, actions, qTables)

            # dynamic computed inside
            x, v = getNextStates(h1=new_action[0], h2=new_action[1], v=states[1], t=t, x_0=states[0], v_0=states[1])
            if np.abs(x) > 2 or np.abs(v) > 1: break  # the ball has fallen

            r = reward(x, v)
            new_states = (np.round(x, decimals=1), np.round(v, decimals=1))
            new_states = checkstates(new_states, qTables)  # check if the states have a match in the discrete grid
            qTables = hysteretic(qTables, states, actions, alpha, beta, r, gamma, new_states)
            states = new_states
    print(qTables[0])
    countNot0(qTables)


trainDistributed()
