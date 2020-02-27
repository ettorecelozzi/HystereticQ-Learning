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
            # discrete_states = min(keys, key=lambda x: euclidean(x, states))
            discrete_states = min(keys, key=lambda x: tuple_distance(x, states))
            discrete = True
    return discrete_states if discrete else states


def tuple_distance(tuple1, tuple2):
    """
    Faster distances. Squared distances to speed up the calculation
    :param tuple1: first tuple
    :param tuple2: second tuple
    :return: Squared distances.
    """
    distance = 0
    for t1, t2 in zip(tuple1, tuple2):
        distance += (t1 - t2) ** 2
    return distance


def check_actions(new_actions, actions):
    """
    Verify if the new action have a match in the discrete grid
    :param new_actions: tuple of action to verify
    :param actions: list of all discrete action available
    :return: discrete actions if needed
    """
    discrete_actions = [0.0] * len(new_actions)
    discrete = False
    for a in range(len(new_actions)):
        if new_actions[a] not in actions:
            discrete_actions[a] = min(actions, key=lambda x: abs(x - new_actions[a]))
            discrete = True
    return discrete_actions if discrete else new_actions


def countNot0(qTables):
    """
    Count the number of element that are not 0 in the QTables
    :param qTables: list of the qTables
    :return: number of elment != 0
    """
    counter = [0] * len(qTables)
    qTable_index = 0
    for q in qTables:
        for state in q:
            for a in q[state]:
                if q[state][a] != 0.0:
                    counter[qTable_index] += 1
        qTable_index += 1
    print(counter)
