def distributed(qTables, r, states, actions, alpha):
    """
    Eq. 2 Hysteretic Q-Learning paper.
    qTables: Qtables of the agents
    r: reward
    states: states as QTable index
    actions: actions chosen
    alpha: learning rate
    :return: list of qtables updated
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
    actions: actions chosen
    r: reward
    gamma: discount factor
    alpha: learning rate
    qTable: single qTable of the "central" agent. Shape = (100,50,15,15)
    new_states: new states
    :return: qTable updated
    """
    maximum = 0 if not qTable[new_states] else max(qTable[new_states].values())
    qTable[states][actions] = \
        (1 - alpha) * qTable[states][actions] + alpha * (r + gamma * maximum)
    return qTable


def decentralized(qTables, states, actions, alpha, r, gamma, new_states):
    """
    Eq. 5 Hysteretic Q-Learning paper
    qTables: list of the qTables, one for each agent. Shape = (100,50,15)
    states: state index for the qTable
    actions: actions chosen
    r: reward
    gamma: discount factor
    alpha: learning rate
    new_states: new states
    :return: list of qTables updated
    """
    for a, q in zip(actions, qTables):
        maximum = 0 if not q[new_states] else max(q[new_states].values())
        q[states][a] = (1 - alpha) * q[states][a] + alpha * (r + gamma * maximum)
    return qTables


def hysteretic(qTables, states, actions, alpha, beta, r, gamma, new_states):
    """
    Eq. 6 Hysteretic Q-Learning paper
    qTables: list of the qTables, one for each agent. Shape = (100,50,15)
    states: state index for QTables
    actions: actions chosen
    alpha: first learning rate
    beta: second learning rate
    r: reward
    gamma: discount factor
    new_states: new states
    :return: list of qTables updated
    """
    for a, q in zip(actions, qTables):
        maximum = 0 if not q[new_states] else max(q[new_states].values())
        delta = r + gamma * maximum - q[states][a]
        if delta >= 0:
            q[states][a] += delta * alpha
        else:
            q[states][a] += delta * beta
    return qTables
