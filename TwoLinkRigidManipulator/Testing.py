import matplotlib.pyplot as plt
import pickle as pkl
from TwoLinkRigidManipulator.EnvironmentRigidManipulator import *
from Utility import floatDD  # do not delete, pickle need it!


def testTwoLinkRigidManipulator(qTables, algorithm, centralized=False):
    """
    Verify how much the robots have learnt.
    :param qTables: trained Q-Tables
    :param algorithm: Q-Algorithm used
    """
    outputAngle1, outputSpeed1 = [], []
    outputAngle2, outputSpeed2 = [], []
    states = (-1.15, -3.2, 0, 0)
    outputAngle1.append(states[0])
    outputAngle2.append(states[1])
    outputSpeed1.append(states[2])
    outputSpeed2.append(states[3])
    for i in range(300):
        actions = choose_action(states, None, qTables, None, centralized)
        if i >20:
            a=1
        theta1, v1 = getNextTheta1States(tau1=actions[0], tau2=actions[1], theta1=states[0],
                                         theta2=states[1], v1=states[2], t=0.03)
        outputAngle1.append(theta1)
        outputSpeed1.append(v1)
        theta2, v2 = getNextTheta2States(theta2=states[1], t=0.03, tau2=actions[1], v2=states[3])
        outputAngle2.append(theta2)
        outputSpeed2.append(v2)

        new_states = (theta1, theta2, v1, v2)
        new_states = check_states(new_states)
        states = new_states
    plt.plot(outputAngle1, '-', label="Space1")
    plt.plot(outputSpeed1, '-', label="Speed1")
    plt.plot(outputAngle2, '-', label="Space2")
    plt.plot(outputSpeed2, '-', label="Speed2")
    plt.legend()
    plt.title(algorithm)
    plt.savefig('./Plots/' + algorithm + '_test.png')
    plt.clf()

#
# for algorithm in ['Distributed', 'Decentralized', 'Hysteretic']:
#     with open('./QTables/qT1_' + algorithm + '.p', 'rb') as file:
#         qTable1 = pkl.load(file)
#     with open('./QTables/qT2_' + algorithm + '.p', 'rb') as file:
#         qTable2 = pkl.load(file)
#     qTables = [qTable1, qTable2]
#     testTwoLinkRigidManipulator(qTables, algorithm)

algorithm = 'Centralized'
with open('./QTables/qT_' + algorithm + '.p', 'rb') as file:
    qTable = pkl.load(file)
testTwoLinkRigidManipulator(qTable, algorithm, centralized=True)
