import numpy as np
from scipy.spatial.distance import euclidean


def generateQTable():
    """
    Create qtable for the agent
    :return: QTable
    """
    dict = {}
    positions = np.round(list(np.linspace(-2, 2, 100)), decimals=1)
    velocities = np.round(list(np.linspace(-3, 3, 50)), decimals=1)
    actions = np.round(list(np.linspace(-1, 1, 15)), decimals=1)
    for p in positions:
        for v in velocities:
            dict[(p, v)] = {}
            for a in actions:
                dict[(p, v)][a] = 0.0
    return dict


def checkstates(states, qTables):
    """
    Verify if the new states belong to the discrete grid
    :param states: float states
    :param qTables: QTable of the agent. Two for decentralized
    :return: states in the discrete grid
    """
    for q in qTables:
        keys = list(q.keys())
        if states not in keys:
            discrete_states = min(keys, key=lambda x: euclidean(x, states))
            return discrete_states
    return states


def countNot0(qTables):
    counter = [0] * len(qTables)
    qTable_index = 0
    for q in qTables:
        for state in q:
            for a in q[state]:
                if q[state][a] != 0.0:
                    counter[qTable_index] += 1
        qTable_index += 1
    print(counter)
