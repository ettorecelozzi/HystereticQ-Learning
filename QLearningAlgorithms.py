from Environment import *


def distributed(qTables, r, x, v, actions, alpha):
    """
    Eq. 2 Hysteretic Q-Learning paper.
    qTables: Qtables of the agents
    r: reward
    x: first state Qtable index
    v: second state Qtable index
    actions: actions available
    alpha: learning rate
    :return: the qtables updated
    """
    for a, q in zip(actions, qTables):
        delta = r - q[x, v, a]
        if delta >= 0:
            q[x, v, a] += alpha * delta
    return qTables


def centralized(x, v, actions, r, gamma, alpha, qTable, states):
    """
    Eq. 4 Hysteretic Q-Learning paper
    x: first state index
    v: second state index
    actions: all possible actions. Shape = (15,)
    r: reward
    gamma: discount factor
    alpha: learning rate
    qTable: single qTable of the "central" agent. Shape = (100,50,15,15)
    states: new states
    :return: qTable updated
    """
    qTable[x, v, actions] = \
        (1 - alpha) * qTable[x, v, actions] + alpha * [r + gamma * (np.max(qTable[states[0], states[1]]))]
    return qTable


def decentralized(qTables, x, v, actions, alpha, r, gamma, states):
    """
    Eq. 5 Hysteretic Q-Learning paper
    qTables: list of the qTables, one for each agent. Shape = (100,50,15)
    x: first state index
    v: second state index
    actions: all possible actions. Shape = (15,)
    r: reward
    gamma: discount factor
    alpha: learning rate
    states: new states
    :return: qTables updated
    """
    # states = getNextStates(h1, h2, v, t, x_0, v_0) # this to discover the actions of the new states
    for a, q in zip(actions, qTables):
        q[x, v, a] = (1 - alpha) * q[x, v, a] + alpha * [r + gamma * (np.max(q[states[0], states[1]]))]
    return qTables


def hysteretic(qTables, x, v, actions, alpha, beta, r, gamma, states):
    """
    Eq. 6 Hysteretic Q-Learning paper
    qTables: list of the qTables, one for each agent. Shape = (100,50,15)
    x: first state index
    v: second state index
    actions: all possible actions. Shape = (15,)
    alpha: first learning rate
    beta: second learning rate
    r: reward
    gamma: discount factor
    states: new states
    :return: qTables updated
    """
    for a, q in zip(actions, qTables):
        delta = r + gamma * np.max(q[states[0], states[1]]) - q[x, v, a]
        if delta >= 0:
            q[x, v, a] += delta * alpha
        else:
            q[x, v, a] += delta * beta
    return qTables
