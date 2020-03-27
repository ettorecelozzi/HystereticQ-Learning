import numpy as np
from Utility import *
from scipy.spatial.distance import euclidean

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
g1 = (m1 * c1 + m2 * l1) * g
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


def gravityVector(theta):
    m = np.zeros(shape=(2, 1))
    m[0][0] = -g1 * np.sin(theta[0]) - g2 * np.sin((theta[0] + theta[1]))
    m[1][0] = -g2 * np.sin((theta[0] + theta[1]))
    return m


def dynamic(thetas, speeds, taus):
    """
    4th order dynamic without assumptions like gravity=0 etc.
    :param thetas: matrix of the angles. shape=(2x1)
    :param speeds: matrix of the speeds. shape=(2x1)
    :param taus: matrix of the actions. shape=(2x1)
    :return: angles accelaration. shape=(1x2)
    """
    cc = coriolisMatrix(thetas, speeds)
    g = gravityVector(thetas)
    m = massMatrix(thetas)
    num = np.transpose(taus - np.dot(cc, thetas) + g)
    ret = np.dot(num, np.linalg.inv(m))
    return ret.flatten()


def getNexstates(thetas, speeds, taus):
    """
    Compute the new states given the previous states and the actions
    :param thetas: matrix of the angles. shape=(2x1)
    :param speeds: matrix of the speeds. shape=(2x1)
    :param taus: matrix of the actions. shape=(2x1)
    :return: new states: theta1, theta2, v1, v2
    """
    a = dynamic(thetas, speeds, taus)
    new_theta1 = thetas[0] + 0.03 * speeds[0]
    new_v1 = speeds[0] + 0.03 * a[0]
    new_theta2 = thetas[1] + 0.03 * speeds[1]
    new_v2 = speeds[1] + 0.03 * a[1]
    return new_theta1, new_theta2, new_v1, new_v2


def dynamictheta1(tau1, tau2, theta2, v1):
    """
    Get the acceleration for the first joint
    :param tau1: torque of the first joint
    :param tau2: torque of the second joint
    :param theta2: speed of the second joint
    :param v1: speed of the first joint
    :return: acceleration
    """
    a1 = (1 / (p2 * (p1 + p2 + 2 * p3 * np.cos(theta2)))) * (
            p2 * (tau1 - b1 * v1) - ((p2 + p3 * np.cos(theta2)) * tau2))
    return a1


def dynamictheta2(tau2, v2):
    """
    Get the acceleration for the second joint
    :param tau2: torque of the second joint
    :param v2: speed of the second joint
    :return: acceleration
    """
    a2 = (tau2 / p2) - (b2 * v2)
    return a2


def getNextTheta1States(tau1, tau2, theta1, theta2, t, v1):
    """
    return the next states of the first joint given the acceleration
    :param tau1: torque of the first joint
    :param tau2: torque of the second joint
    :param theta1: speed of the first joint
    :param theta2: speed of the second joint
    :param t: elapsed time
    :param v1: speed of the first joint
    :return: new speed and new speed
    """
    a1 = dynamictheta1(tau1, tau2, theta2, v1)
    new_theta1 = (theta1 + t * v1)
    # Ignore multiples of 2*pi
    if new_theta1 < -2 * np.pi:
        new_theta1 = new_theta1 % (2 * np.pi)
    elif new_theta1 > 2 * np.pi:
        new_theta1 = new_theta1 % (2 * np.pi)
    new_v1 = v1 + t * a1
    return new_theta1, new_v1


def getNextTheta2States(theta2, t, tau2, v2):
    """
    return the next states of the second joint given the acceleration
    :param tau2: torque of the second joint
    :param theta2: speed of the second joint
    :param t: elapsed time
    :param v2: speed of the second joint
    :return: new speed and new speed
    """
    a2 = dynamictheta2(tau2, v2)
    new_theta2 = (theta2 + t * v2)
    # Ignore multiples of 2*pi
    if new_theta2 < -2 * np.pi:
        new_theta2 = new_theta2 % (2 * np.pi)
    elif new_theta2 > 2 * np.pi:
        new_theta2 = new_theta2 % (2 * np.pi)
    new_v2 = v2 + t * a2
    return new_theta2, new_v2


def reward(angle, velocity):
    """
    Reward function as defined in the paper
    :param angle: first state (space)
    :param velocity: second state (derivate of the state (speed))
    :return: the reward with respect of the states
    """
    if np.abs(angle) <= 5 * (np.pi / 180) and np.abs(velocity) <= 0.1:
        return 0
    else:
        return -0.5


def rewardQuadratic(theta, v):
    """
    Quadratic reward
    :param theta: space
    :param v: speed
    :return: reward associated to the space and the speed
    """
    return - np.power(theta, 2) - 0.05 * np.power(v, 2)


def rewardCentralized(angles, velocities):
    """
    Reward function for the centralized case as defined in the paper
    :param angles: angles states
    :param velocities: velocities states
    :return: reward of the states
    """
    if np.abs(angles[0]) <= 5 * (np.pi / 180) and np.abs(angles[1]) <= 5 * (np.pi / 180) \
            and np.abs(velocities[0]) <= 0.1 and np.abs(velocities[1]) <= 0.1:
        return 0
    else:
        return -0.5


def choose_action(states, actions, qTables, trial, centralized=False, numOfEps=40, trials=5000):
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
    if trial is not None and numOfEps > 0:
        epsilons = np.linspace(0.8, 0.1, numOfEps)
        index = int(trial // (trials / numOfEps))
        eps = epsilons[index]
    else:
        eps = 0.1

    if centralized is False:
        numberOfAgents = len(qTables)
        new_actions = [0] * numberOfAgents
        for q in range(len(qTables)):
            if actions is not None and np.random.uniform() < eps:
                action = np.random.choice(actions[q])
            else:
                maximum = 0 if not qTables[q][states] else max(qTables[q][states].values())
                action = getKeysByValue(qTables[q][states], maximum)
            new_actions[q] = action
        return new_actions
    else:
        if actions is not None and np.random.uniform() < eps:
            new_actions = [np.random.choice(actions[0]), np.random.choice(actions[1])]  # [tau1,tau2]
        else:
            maximum = 0.0 if not qTables[states] else max(qTables[states].values())
            new_actions = getKeysByValue(qTables[states], maximum, centralized)
        return tuple(new_actions)


def interpolate_continuous_states(states, decimals):
    """
    Givent the continuous states return the states given by the interpolation of fixed centers
    :param states: continuous states
    :return: discrete states
    """
    theta1, theta2, v1, v2 = states[0], states[1], states[2], states[3]

    angles = np.array([-180 * np.pi / 180, -130 * np.pi / 180, -80 * np.pi / 180, -30 * np.pi / 180, -15 * np.pi / 180,
                       -5 * np.pi / 180, 0,
                       5 * np.pi / 180, 15 * np.pi / 180, 30 * np.pi / 180, 80 * np.pi / 180, 130 * np.pi / 180])

    speeds = np.array([-360 * np.pi / 180, -180 * np.pi / 180, -30 * np.pi / 180, 0, 30 * np.pi / 180,
                       180 * np.pi / 180, 360 * np.pi / 180])

    # cyclic angles, from -pi to positive_extr
    positive_extr = (130 * np.pi / 180)
    if theta1 > positive_extr: theta1 = -np.pi + np.abs(theta1 % positive_extr)
    if theta1 < -np.pi: theta1 = positive_extr - np.abs(theta1 % -np.pi)
    if theta2 > positive_extr: theta2 = -np.pi + np.abs(theta2 % positive_extr)
    if theta2 < -np.pi: theta2 = positive_extr - np.abs(theta2 % -np.pi)

    index_theta1 = min(range(len(angles)), key=lambda i: euclidean(angles[i], states[0]))
    index_theta2 = min(range(len(angles)), key=lambda i: euclidean(angles[i], states[1]))
    index_v1 = min(range(len(speeds)), key=lambda i: euclidean(speeds[i], states[2]))
    index_v2 = min(range(len(speeds)), key=lambda i: euclidean(speeds[i], states[3]))

    new_states = (round(getWeights(theta1, angles, index_theta1) * theta1, decimals),
                  round(getWeights(theta2, angles, index_theta2) * theta2, decimals),
                  round(getWeights(v1, speeds, index_v1) * v1, decimals),
                  round(getWeights(v2, speeds, index_v2) * v2, decimals))

    return new_states


# def interpolate_continuous_states(states, decimals):
#     """
#     Givent the continuous states return the states given by the interpolation of fixed centers
#     :param states: continuous states
#     :return: discrete states
#     """
#     theta1, theta2, v1, v2 = states[0], states[1], states[2], states[3]
#
#     angles = np.array([-180, -130, -80, -30, -15, -5, 0, 5, 15, 30, 80, 130]) * np.pi / 180
#     speeds = np.array([-360, -180, -30, 0, 30, 180, 360]) * np.pi / 180
#
#     # cyclic angles, from -pi to positive_extr
#     positive_extr = (130 * np.pi / 180)
#     if theta1 > positive_extr: theta1 = -np.pi + np.abs(theta1 % positive_extr)
#     if theta1 < -np.pi: theta1 = positive_extr - np.abs(theta1 % -np.pi)
#     if theta2 > positive_extr: theta2 = -np.pi + np.abs(theta2 % positive_extr)
#     if theta2 < -np.pi: theta2 = positive_extr - np.abs(theta2 % -np.pi)
#
#     index_theta1 = min(range(len(angles)), key=lambda i: euclidean(angles[i], states[0]))
#     index_theta2 = min(range(len(angles)), key=lambda i: euclidean(angles[i], states[1]))
#     index_v1 = min(range(len(speeds)), key=lambda i: euclidean(speeds[i], states[2]))
#     index_v2 = min(range(len(speeds)), key=lambda i: euclidean(speeds[i], states[3]))
#
#     new_states = (np.round(getWeights(theta1, angles, index_theta1) * theta1, decimals=decimals),
#                   np.round(getWeights(theta2, angles, index_theta2) * theta2, decimals=decimals),
#                   np.round(getWeights(v1, speeds, index_v1) * v1, decimals=decimals),
#                   np.round(getWeights(v2, speeds, index_v2) * v2, decimals=decimals))
#     return new_states


def getWeights(state, centers, index):
    """
    Compute the weight for each state
    :param state: float
    :param centers: list of centers
    :param index: index of the closest discrete state
    :return: weight
    """
    weight = -1
    if index == 0:
        weight = max(0, (centers[index + 1] - state) / (centers[index + 1] - centers[index]))
    elif 0 < index < len(centers) - 1:
        weight = max(0, min((state - centers[index - 1]) / (centers[index] - centers[index - 1]),
                            (centers[index + 1] - state) / (centers[index + 1] - centers[index])))
    elif index == len(centers) - 1:
        weight = max(0, (state - centers[index - 1]) / (centers[index] - centers[index - 1]))
    if weight == -1: raise Exception('Weight error')
    return weight


def check_states(states):
    """
    Verify if the new states belong to the discrete grid, otherwise return the closest discrete value in the grid
    :param states: float states
    :return: states in the discrete grid
    """
    states = list(states)

    # cyclic angles, from -pi to pi
    if states[0] > np.pi: states[0] = -np.pi + np.abs(states[0] % np.pi)
    if states[0] < -np.pi: states[0] = np.pi - np.abs(states[0] % -np.pi)
    if states[1] > np.pi: states[1] = -np.pi + np.abs(states[1] % np.pi)
    if states[1] < -np.pi: states[1] = np.pi - np.abs(states[1] % -np.pi)

    dim1 = 100
    dim2 = 100
    angles = np.round(list(np.linspace(- np.pi, np.pi, dim1)), decimals=2)
    velocities = np.round(list(np.linspace(-2 * np.pi, 2 * np.pi, dim2)), decimals=2)
    angle_space = np.pi / (dim1 // 2)
    velocity_space = 2 * np.pi / (dim2 // 2)
    angle_index1 = np.abs(int(np.round(states[0] / angle_space, decimals=0)))
    angle_index2 = np.abs(int(np.round(states[1] / angle_space, decimals=0)))
    velocity_index1 = np.abs(int(np.round(states[2] / velocity_space, decimals=0)))
    velocity_index2 = np.abs(int(np.round(states[3] / velocity_space, decimals=0)))
    if states[0] > 0:
        angle_index1 += (dim1 // 2)
    else:
        angle_index1 = (dim1 // 2) - angle_index1

    if states[1] > 0:
        angle_index2 += (dim1 // 2)
    else:
        angle_index2 = (dim1 // 2) - angle_index2

    if states[2] > 0:
        velocity_index1 += (dim2 // 2)
    else:
        velocity_index1 = (dim2 // 2) - velocity_index1

    if states[3] > 0:
        velocity_index2 += (dim2 // 2)
    else:
        velocity_index2 = (dim2 // 2) - velocity_index2

    ##### P1 #####
    if angle_index1 != dim1 and angle_index1 != 0:
        if angle_index1 == (dim1 - 1):
            possible_angles1 = [angles[angle_index1 - 1], angles[angle_index1]]
        else:
            possible_angles1 = [angles[angle_index1 - 1], angles[angle_index1],
                                angles[angle_index1 + 1]]
    elif angle_index1 == dim1:
        possible_angles1 = [angles[angle_index1 - 1], angles[angle_index1 - 2]]
    else:
        possible_angles1 = [angles[angle_index1], angles[angle_index1 + 1]]

    ##### P2 #####
    if angle_index2 != dim1 and angle_index2 != 0:
        if angle_index2 == (dim1 - 1):
            possible_angles2 = [angles[angle_index2 - 1], angles[angle_index2]]
        else:
            possible_angles2 = [angles[angle_index2 - 1], angles[angle_index2],
                                angles[angle_index2 + 1]]
    elif angle_index2 == dim1:
        possible_angles2 = [angles[angle_index2 - 1], angles[angle_index2 - 2]]
    else:
        possible_angles2 = [angles[angle_index2], angles[angle_index2 + 1]]

    ##### V1 #####
    if velocity_index1 != dim2 and velocity_index1 != 0:
        if velocity_index1 == (dim2 - 1):
            possible_velocities1 = [velocities[velocity_index1 - 1], velocities[velocity_index1]]
        else:
            possible_velocities1 = [velocities[velocity_index1 - 1], velocities[velocity_index1],
                                    velocities[velocity_index1 + 1]]
    elif velocity_index1 == dim2:
        possible_velocities1 = [velocities[velocity_index1 - 1], velocities[velocity_index1 - 2]]
    else:
        possible_velocities1 = [velocities[velocity_index1], velocities[velocity_index1 + 1]]

    ##### V2 #####
    if velocity_index2 != dim2 and velocity_index2 != 0:
        if velocity_index2 == (dim2 - 1):
            possible_velocities2 = [velocities[velocity_index2 - 1], velocities[velocity_index2]]
        else:
            possible_velocities2 = [velocities[velocity_index2 - 1], velocities[velocity_index2],
                                    velocities[velocity_index2 + 1]]
    elif velocity_index2 == dim2:
        possible_velocities2 = [velocities[velocity_index2 - 1], velocities[velocity_index2 - 2]]
    else:
        possible_velocities2 = [velocities[velocity_index2], velocities[velocity_index2 + 1]]

    new_states = (min(possible_angles1, key=lambda x: abs(x - states[0])),
                  min(possible_angles2, key=lambda x: abs(x - states[1])),
                  min(possible_velocities1, key=lambda x: abs(x - states[2])),
                  min(possible_velocities2, key=lambda x: abs(x - states[3])))
    return new_states
