import numpy as np
from scipy.spatial.distance import euclidean


def generateQTable():
    """
    Create qtable for the agent
    :return: QTable
    """
    dict = {}
    positions = np.round(list(np.linspace(-1, 1, 100)), decimals=1)
    velocities = np.round(list(np.linspace(-3, 3, 50)), decimals=1)
    actions = np.round(list(np.linspace(-1, 1, 15)), decimals=1)
    for p in positions:
        for v in velocities:
            dict[(p, v)] = {}
            for a in actions:
                dict[(p, v)][a] = 0.0
    return dict


def check_states(states, qTables):
    """
    Verify if the new states belong to the discrete grid
    :param states: float states
    :param qTables: QTable of the agent. Two for decentralized
    :return: states in the discrete grid
    """
    discrete_states = -1
    discrete = False
    for q in range(len(qTables)):
        keys = list(qTables[q].keys())
        if states not in keys:
            discrete_states = min(keys, key=lambda x: euclidean(x, states))
            discrete = True
    return discrete_states if discrete else states


def check_actions(new_actions, actions):
    """

    :param new_actions:
    :param actions:
    :return:
    """
    discrete_actions = [0.0] * len(new_actions)
    discrete = False
    # print(new_actions)
    for a in range(len(new_actions)):
        if new_actions[a] not in actions:
            discrete_actions[a] = min(actions, key=lambda x: abs(x - new_actions[a]))
            discrete = True
    # print('disc')
    # print(discrete_actions)
    return discrete_actions if discrete else new_actions


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
