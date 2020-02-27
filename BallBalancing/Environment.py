import numpy as np


def generateQTable():
    """
    Create qtable for the agent
    :return: QTable
    """
    dict = {}
    positions = np.round(list(np.linspace(-1, 1, 100)), decimals=2)
    velocities = np.round(list(np.linspace(-3, 3, 50)), decimals=2)
    actions = np.round(list(np.linspace(-1, 1, 15)), decimals=2)
    for p in positions:
        for v in velocities:
            dict[(p, v)] = {}
            for a in actions:
                dict[(p, v)][a] = 0.0
    return dict


def reward(x, v):
    """
    Reward function
    :param x: first state (space)
    :param v: second state (derivate of the state (speed))
    :return: the reward with respect of the states
    """
    return 0.8 * np.exp(-(np.power(x, 2) / np.power(0.25, 2))) + 0.2 * np.exp(-(np.power(v, 2) / np.power(0.25, 2)))


def dynamic(h1, h2, v):
    """
    Dynamic of the system
    h1: height action 1
    h2: height action 2
    v: speed state
    :return: acceleration
    """
    m = 0.5
    g = 9.8
    l = 2
    c = 0.01
    return ((-c * v) + (m * g * ((h1 - h2) / l))) / m


def newPos(a, t, v, x_0):
    """
    Compute the new position.
    :param a: acceleration
    :param t: time
    :param v: speed
    :param x_0: starting position
    :return: new position
    """
    return 1 / 2 * a * t * t + v * t + x_0


def newSpeed(a, t, v_0):
    """
    Compute the new speed
    :param a: acceleration
    :param t: time
    :param v_0: starting speed
    :return: new speed
    """
    return a * t + v_0


def getNextStates(h1, h2, v, t, x_0, v_0):
    """
    Compute the new states
    :param h1: height 1
    :param h2: height 2
    :param v: speed
    :param t: time
    :param x_0: initial position
    :param v_0: initial speed
    :return: new states
    """
    a = dynamic(h1, h2, v)
    xnew = newPos(a, 0.3, v, x_0)
    vnew = newSpeed(a, 0.3, v_0)
    return xnew, vnew


def choose_action(states, actions, qTables, epsilon=0.1):
    """
    Select next action balancing exploration and exploitation
    :param states: actual states
    :param actions: indexes of the actions
    :param qTables: QTable
    :param epsilon: parameter used to adopt exploration/exploitation
    :return: the new actions to perform
    """
    numberOfAgents = len(qTables)
    new_actions = [0] * numberOfAgents
    for q in range(len(qTables)):
        if np.random.uniform() < epsilon:
            action = np.random.choice(actions)
        else:
            action = getKeysByValue(qTables[q][states],max(qTables[q][states].values()))
        new_actions[q] = action
    return new_actions

def getKeysByValue(dictOfElements, valueToFind):
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            return item[0]
