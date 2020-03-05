import numpy as np

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
    return [new_theta1, new_v1]


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
    return [new_theta2, new_v2]


def generateQTables(centralized=False):
    """

    :param centralized:
    :return:
    """
    dict = {}
    angles = np.round(list(np.linspace(-2 * np.pi, 2 * np.pi, 50)))
    speeds = np.round(list(np.linspace(-2 * np.pi, 2 * np.pi, 50)))
    tau1 = np.round(list(np.linspace(-0.2, 0.2, 50)))
    tau2 = np.round(list(np.linspace(-0.1, 0.1, 50)))