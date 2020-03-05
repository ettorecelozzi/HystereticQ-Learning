import numpy as np
from Utility import *

g = 9.81
l1 = l2 = 0.1
m1 = 1.25
m2 = 1
i1 = 0.004
i2 = 0.003
c1 = c2 = 0.05
b1 = 0.1
b2 = 0.02
p1 = m1 * c1 * c1 + m2 * l1 * l1 + i1
p2 = m2 * c2 * c2 + i2
p3 = m2 * l1 * c2
g1 = (m1 * c1 + m2 * c1) * g
g2 = m2 * c2 * g


def massMatrix(theta):
    m = np.zeros(shape=(2, 2))
    m[0][0] = p1 + p2 + p3 * (np.cos(theta[1]))
    m[0][1] = p2 + p3 * (np.cos(theta[1]))
    m[1][0] = p2 + p3 * (np.cos(theta[1]))
    m[1][1] = p2
    return m


def coriolisMatrix(theta, theta_point):
    m = np.zeros(shape=(2, 2))
    m[0][0] = b1 - p3 * theta_point[1] * (np.sin(theta[1]))
    m[0][1] = - p3 * (theta_point[0] + theta_point[1]) * (np.sin(theta[1]))
    m[1][0] = - p3 * (theta_point[0] + theta_point[1]) * (np.sin(theta[1]))
    m[1][1] = b2
    return m


def centrifugalMatrix(theta):
    m = np.zeros(shape=(2, 1))
    m[0][0] = -g1 * np.sin(theta[0]) - g2 * np.sin((theta[0] + theta[1]))
    m[1][0] = -g2 * np.sin((theta[0] + theta[1]))
    return m


def dynamictheta1(tau1, tau2, theta2, v1):
    """

    :param tau1:
    :param tau2:
    :param theta1:
    :param theta2:
    :param v1:
    :param v2:
    :return:
    """
    a1 = 1 / (p2 * (p1 + p2 + 2 * p3 * np.cos(theta2))) * (p1 * (tau1 - b1 * v1) - (p2 + p3 * np.cos(theta2) * tau2))
    return a1


def dynamictheta2(tau2, v2):
    """

    :param tau2:
    :param v2:
    :return:
    """
    a2 = tau2 / p2 - b2 * v2
    return a2


def getNextTheta1States(tau1, tau2, theta1, theta2, t, v1):
    """

    :param tau1:
    :param tau2:
    :param theta1:
    :param theta2:
    :param t:
    :param v1:
    :param v2:
    :return:
    """
    a1 = dynamictheta1(tau1, tau2, theta2, v1)
    new_theta1 = theta1 + t * v1
    new_v1 = v1 + t * a1
    return new_theta1, new_v1


def getNextTheta2States(theta2, t, tau2, v2):
    """

    :param theta2:
    :param t:
    :param tau2:
    :param v2:
    :return:
    """
    a2 = dynamictheta2(tau2, v2)
    new_theta2 = theta2 + t * v2
    new_v2 = v2 + t * a2
    return new_theta2, new_v2


def generateQTables(centralized=False):
    """

    :param centralized:
    :return:
    """
    angles = np.round(list(np.linspace(-2 * np.pi, 2 * np.pi, 50)), decimals=2)
    speeds = np.round(list(np.linspace(-2 * np.pi, 2 * np.pi, 50)), decimals=2)
    tau1 = np.round(list(np.linspace(-0.2, 0.2, 50)), decimals=2)
    tau2 = np.round(list(np.linspace(-0.1, 0.1, 50)), decimals=2)
    if centralized is False:
        qTable1 = {}
        qTable2 = {}
        for ang1 in angles:  # theta1
            for ang2 in angles:  # theta2
                for spd in speeds:
                    qTable2[(ang2, spd)] = {}
                    qTable1[(ang1, ang2, spd)] = {}
                    for t1 in tau1:
                        for t2 in tau2:
                            qTable1[(ang1, ang2, spd)][(t1, t2)] = 0.0
                            qTable2[(ang2, spd)][t2] = 0.0
        return [qTable1, qTable2]
    else:
        qTable = {}
        for ang1 in angles:  # theta1
            for ang2 in angles:  # theta2
                for spd1 in speeds:  # speed1
                    for spd2 in speeds:  # speed2
                        qTable[(ang1, ang2, spd1, spd2)] = {}
                        for t1 in tau1:
                            for t2 in tau2:
                                qTable[(ang1, ang2, spd1, spd2)][(t1, t2)] = 0.0
        return qTable

def reward(angle, velocity, centralized = False):
    """
    Reward function
    :param x: first state (space)
    :param v: second state (derivate of the state (speed))
    :return: the reward with respect of the states
    """
    if centralized is False:
        if np.abs(angle) <= 5*(np.pi/180) and np.abs(velocity) <= 0.1:
            return 0
        else:
            return -0.5
    else:
        if (np.abs(angle[0]) <= 5 * (np.pi / 180) and np.abs(velocity[0]) <= 0.1) and (np.abs(angle[1]) <= 5 * (np.pi / 180) and np.abs(velocity[1]) <= 0.1):
            return 0
        else:
            return -0.5

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
                action = np.random.choice(actions[q])
            else:
                action = getKeysByValue(qTables[q][states], max(qTables[q][states].values()))
            new_actions[q] = action
        return new_actions
    else:
        if np.random.uniform() < eps:
            new_actions = [np.random.choice(actions[0]), np.random.choice(actions[1])]  # [tau1,tau2]
        else:
            new_actions = getKeysByValue(qTables[states], max(qTables[states].values()))
        return tuple(new_actions)
def check_states(states):
    """
    Verify if the new states belong to the discrete grid, otherwise return the closest discrete value in the grid
    :param states: float states
    :return: states in the discrete grid
    """
    angles = np.round(list(np.linspace(-2 * np.pi, 2 * np.pi, 50)), decimals=2)
    velocities = np.round(list(np.linspace(-2 * np.pi, 2 * np.pi, 50)), decimals=2)
    angle_space = 2 / 25
    velocity_space = 2 / 25
    angle_index = np.abs(int(np.round(states[0] / angle_space, decimals=0)))
    velocity_index = np.abs(int(np.round(states[1] / velocity_space, decimals=0)))
    if states[0] > 0:
        angle_index += 25
    else:
        angle_index = 25 - angle_index
    if states[1] > 0:
        velocity_index += 25
    else:
        velocity_index = 25 - velocity_index

    if angle_index != 50 and angle_index != 0:
        if angle_index == 49:
            possible_positions = [angles[angle_index - 1], angles[angle_index]]
        else:
            possible_positions = [angles[angle_index - 1], angles[angle_index],
                                  angles[angle_index + 1]]
    elif angle_index == 50:
        possible_positions = [angles[angle_index - 1], angles[angle_index - 2]]
    else:
        possible_positions = [angles[angle_index], angles[angle_index + 1]]

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