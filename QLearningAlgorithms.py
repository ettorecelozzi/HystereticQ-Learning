from Environment import *


def distributed(qTables, r, states, actions, alpha):
    """
    Eq. 2 Hysteretic Q-Learning paper.
    qTables: Qtables of the agents
    r: reward
    states: states as QTable index
    actions: actions available
    alpha: learning rate
    :return: the qtables updated
    """
    for a, q in zip(actions, qTables):
        delta = r - q[states][a]
        if delta >= 0:
            q[states][a] += alpha * delta
    return qTables


def centralized(states, actions, r, gamma, alpha, qTable, new_states):
    """
    Eq. 4 Hysteretic Q-Learning paper
    x: first state index
    v: second state index
    actions: all possible actions. Shape = (15,)
    r: reward
    gamma: discount factor
    alpha: learning rate
    qTable: single qTable of the "central" agent. Shape = (100,50,15,15)
    new_states: new states
    :return: qTable updated
    """
    qTable[states][actions] = \
        (1 - alpha) * qTable[states][actions] + alpha * [r + gamma * (np.max(qTable[new_states]))]
    return qTable


def decentralized(qTables, states, actions, alpha, r, gamma, new_states):
    """
    Eq. 5 Hysteretic Q-Learning paper
    qTables: list of the qTables, one for each agent. Shape = (100,50,15)
    x: first state index
    v: second state index
    actions: all possible actions. Shape = (15,)
    r: reward
    gamma: discount factor
    alpha: learning rate
    new_states: new states
    :return: qTables updated
    """
    # states = getNextStates(h1, h2, v, t, x_0, v_0) # this to discover the actions of the new states
    for a, q in zip(actions, qTables):
        q[states][a] = (1 - alpha) * q[states][a] + alpha * [r + gamma * (np.max(q[new_states]))]
    return qTables


def hysteretic(qTables, states, actions, alpha, beta, r, gamma, new_states):
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
    new_states: new states
    :return: qTables updated
    """
    for a, q in zip(actions, qTables):
        delta = r + gamma * np.max(q[new_states]) - q[states][a]
        if delta >= 0:
            q[states][a] += delta * alpha
        else:
            q[states][a] += delta * beta
    return qTables
