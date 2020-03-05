from TwoLinkRigidManipulator.EnvironmentRigidManipulator import *
from Utility import *
from QLearningAlgorithms import *
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def trainHysteretic():
    # parameters
    gamma = 0.9
    beta = 0.1
    alpha = 0.9

    # generate Q-Table
    qTable1 = defaultdict(lambda: defaultdict(float))
    qTable2 = defaultdict(lambda: defaultdict(float))
    qTables = [qTable1, qTable2]

    tau1 = np.round(list(np.linspace(-0.2, 0.2, 15)), decimals=2)
    tau2 = np.round(list(np.linspace(-0.1, 0.1, 15)), decimals=2)
    actions = [tau1, tau2]
    output20 = []
    for i in range(1):
        output = []
        for trial in range(5000):
            printProgressBar(trial, 5000, prefix='Iteration: ' + str(i))
            states = (-1.15, -3.2, 0, 0)
            rewardSum = 0
            for t in np.arange(0, 20, 0.03):
                new_actions = choose_action(states, actions, qTables, trial)

                theta1, v1 = getNextTheta1States(tau1=new_actions[0], tau2=new_actions[1], theta1=states[0],
                                                 theta2=states[1], v1=states[2], t=0.03)
                theta2, v2 = getNextTheta2States(theta2=states[1], t=0.03, tau2=new_actions[1], v2=states[3])

                if v1 > 2 * np.pi: v1 = 2 * np.pi
                if v2 > 2 * np.pi: v2 = 2 * np.pi
                if v1 < -2 * np.pi: v1 = 2 * np.pi
                if v2 < -2 * np.pi: v2 = 2 * np.pi

                r = reward(states[0], states[1])
                rewardSum += r

                new_states = (theta1, theta2, v1, v2)
                new_states = check_states(new_states)

                qTables = hysteretic(qTables, states, new_actions, alpha, beta, r, gamma, new_states)
                states = new_states

            output.append(rewardSum)
        output20.append(output)

    mean_output = np.mean(output20, axis=0)
    plt.scatter(list(range(5000)), mean_output)
    plt.show()

    countNot0(qTables)

    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Hysteretic.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Hysteretic.csv')


def trainCentralized():
    # parameters
    gamma = 0.9
    alpha = 0.9

    # generate Q-Table
    qTable = generateQTables(centralized=True)
    tau1 = np.round(list(np.linspace(-0.2, 0.2, 50)), decimals=2)
    tau2 = np.round(list(np.linspace(-0.1, 0.1, 50)), decimals=2)
    actions = [tau1, tau2]
    output20 = []
    for i in range(1):
        output = []
        for trial in range(5000):
            states = (-1.15, -3.2, 0, 0)
            rewardSum = 0
            for t in np.arange(0, 20, 0.03):
                new_actions = choose_action(states, actions, qTable, trial, centralized=True)

                theta1, v1 = getNextTheta1States(tau1=new_actions[0], tau2=new_actions[1], theta1=states[0],
                                                 theta2=states[1], v1=states[2], t=0.03)
                theta2, v2 = getNextTheta2States(theta2=states[1], t=0.03, tau2=new_actions[1], v2=states[3])

                if v1 > 2 * np.pi: v1 = 2 * np.pi
                if v2 > 2 * np.pi: v2 = 2 * np.pi
                if v1 < -2 * np.pi: v1 = 2 * np.pi
                if v2 < -2 * np.pi: v2 = 2 * np.pi

                r = reward(states[0], states[1])
                rewardSum += r

                new_states = (theta1, theta2, v1, v2)
                new_states = check_states(new_states)

                qTable = centralized(states, new_actions, r, gamma, alpha, qTable, new_states)
                states = new_states

            output.append(rewardSum)
        output20.append(output)

    mean_output = np.mean(output20, axis=0)
    plt.scatter(list(range(5000)), mean_output)
    plt.show()

    countNot0([qTable])

    pd.DataFrame.from_dict(qTable, orient='index').to_csv('./QTables/qT_Centralized.csv')


trainHysteretic()
