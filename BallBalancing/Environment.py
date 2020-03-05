import numpy as np
from Utility import *


def generateQTable(centralized=False):
    """
    Create qtable for the agent
    :param centralized: boolean value to manage the centralized case.
    :return: QTable
    """
    dict = {}
    positions = np.round(list(np.linspace(-1, 1, 100)), decimals=2)
    velocities = np.round(list(np.linspace(-3, 3, 50)), decimals=2)
    actions = np.round(list(np.linspace(-1, 1, 15)), decimals=2)
    if centralized is False:
        for p in positions:
            for v in velocities:
                dict[(p, v)] = {}
                for a in actions:
                    dict[(p, v)][a] = 0.0
        return dict
    else:
        for p in positions:
            for v in velocities:
                dict[(p, v)] = {}
                for a1 in actions:
                    for a2 in actions:
                        dict[(p, v)][(a1, a2)] = 0.0
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
    xnew = newPos(a, t, v, x_0)
    vnew = newSpeed(a, t, v_0)
    return xnew, vnew


def choose_action(states, actions, qTables, trial, centralized=False, numOfEps=20):
    """
    Select next action balancing exploration and exploitation
    :param states: actual states
    :param actions: list of actions available
    :param qTables: QTable
    :param centralized: boolean variable to manage the centralized case
    :param trial: number of the trial (Total 5000)
    :param numOfEps: number of Epsilon. Epsilons are used to adopt exploration/exploitation
    :return: the new actions to perform
    """
    if numOfEps > 0:
        epsilons = np.linspace(0.9, 0.1, numOfEps)
        index = int(trial // (5000 / numOfEps))
        eps = epsilons[index]
    else:
        eps = 0.1

    if centralized is False:
        numberOfAgents = len(qTables)
        new_actions = [0] * numberOfAgents
        for q in range(len(qTables)):
            if np.random.uniform() < eps:
                action = np.random.choice(actions)
            else:
                action = getKeysByValue(qTables[q][states], max(qTables[q][states].values()))
            new_actions[q] = action
        return new_actions
    else:
        if np.random.uniform() < eps:
            new_actions = [np.random.choice(actions), np.random.choice(actions)]
        else:
            new_actions = getKeysByValue(qTables[states], max(qTables[states].values()))
        return tuple(new_actions)


def check_states(states):
    """
    Verify if the new states belong to the discrete grid, otherwise return the closest discrete value in the grid
    :param states: float states
    :return: states in the discrete grid
    """
    positions = np.round(list(np.linspace(-1, 1, 100)), decimals=2)
    velocities = np.round(list(np.linspace(-3, 3, 50)), decimals=2)
    position_space = 1 / 50
    velocity_space = 3 / 25
    position_index = np.abs(int(np.round(states[0] / position_space, decimals=0)))
    velocity_index = np.abs(int(np.round(states[1] / velocity_space, decimals=0)))
    if states[0] > 0:
        position_index += 50
    else:
        position_index = 50 - position_index
    if states[1] > 0:
        velocity_index += 25
    else:
        velocity_index = 25 - velocity_index

    if position_index != 100 and position_index != 0:
        if position_index == 99:
            possible_positions = [positions[position_index - 1], positions[position_index]]
        else:
            possible_positions = [positions[position_index - 1], positions[position_index],
                                  positions[position_index + 1]]
    elif position_index == 100:
        possible_positions = [positions[position_index - 1], positions[position_index - 2]]
    else:
        possible_positions = [positions[position_index], positions[position_index + 1]]

    if velocity_index != 50 and velocity_index != 0:
        if velocity_index == 49:
            possible_velocities = [velocities[velocity_index - 1], velocities[velocity_index]]
        else:
            possible_velocities = [velocities[velocity_index - 1], velocities[velocity_index],
                                   velocities[velocity_index + 1]]
    elif velocity_index == 50:
        possible_velocities = [velocities[velocity_index - 1], velocities[velocity_index - 2]]
    else:
        possible_velocities = [velocities[velocity_index], velocities[velocity_index + 1]]

    new_states = (min(possible_positions, key=lambda x: abs(x - states[0])),
                  min(possible_velocities, key=lambda x: abs(x - states[1])))
    return new_states
