from TwoLinkRigidManipulator.EnvironmentRigidManipulator import *
from Utility import *
from TwoLinkRigidManipulator.QLearningAlgorithms import *
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle as pkl

# parameters
gamma = 0.9
beta = 0.1
alpha = 0.9
tau1 = np.round(list(np.linspace(-0.2, 0.2, 16)), decimals=2)
tau2 = np.round(list(np.linspace(-0.1, 0.1, 16)), decimals=2)
actions = [tau1, tau2]
samplingTime = 0.03


def trainDistributed():
    # generate Q-Table
    qTable1 = defaultdict(floatDD)
    qTable2 = defaultdict(floatDD)
    qTables = [qTable1, qTable2]
    trials = 5000
    for trial in range(trials):
        printProgressBar(trial, trials, prefix='Distributed: ')
        states = (-1.15, -3.2, 0, 0)
        for t in np.arange(0, 20, samplingTime):
            new_actions = choose_action(states, actions, qTables, trial, trials=trials)

            theta1, v1 = getNextTheta1States(tau1=new_actions[0], tau2=new_actions[1], theta1=states[0],
                                             theta2=states[1], v1=states[2], t=samplingTime)
            theta2, v2 = getNextTheta2States(theta2=states[1], t=samplingTime, tau2=new_actions[1], v2=states[3])

            # speed saturation
            if v1 > 2 * np.pi: v1 = 2 * np.pi
            if v2 > 2 * np.pi: v2 = 2 * np.pi
            if v1 < -2 * np.pi: v1 = -2 * np.pi
            if v2 < -2 * np.pi: v2 = -2 * np.pi

            r1 = reward(states[0], states[2])
            r2 = reward(states[1], states[3])
            r = [r1, r2]

            states = (theta1, theta2, v1, v2)
            states = check_states(states)

            qTables = distributed(qTables, r, states, new_actions, alpha)

    countNot0(qTables)

    pkl.dump(qTables[0], open('./QTables/qT1_Distributed.p', 'wb'))
    pkl.dump(qTables[1], open('./QTables/qT2_Distributed.p', 'wb'))
    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Distributed.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Distributed.csv')


def trainDecentralized():
    # generate Q-Table
    qTable1 = defaultdict(floatDD)
    qTable2 = defaultdict(floatDD)
    qTables = [qTable1, qTable2]
    trials = 5000
    for trial in range(trials):
        printProgressBar(trial, trials, prefix='Decentralized: ')
        states = (-1.15, -3.2, 0, 0)
        for t in np.arange(0, 20, 0.03):
            new_actions = choose_action(states, actions, qTables, trial, trials = trials)

            theta1, v1 = getNextTheta1States(tau1=new_actions[0], tau2=new_actions[1], theta1=states[0],
                                             theta2=states[1], v1=states[2], t=0.03)
            theta2, v2 = getNextTheta2States(theta2=states[1], t=samplingTime, tau2=new_actions[1], v2=states[3])

            # speed saturation
            if v1 > 2 * np.pi: v1 = 2 * np.pi
            if v2 > 2 * np.pi: v2 = 2 * np.pi
            if v1 < -2 * np.pi: v1 = -2 * np.pi
            if v2 < -2 * np.pi: v2 = -2 * np.pi

            r1 = reward(states[0], states[2])
            r2 = reward(states[1], states[3])
            r = [r1, r2]

            new_states = (theta1, theta2, v1, v2)
            new_states = check_states(new_states)

            qTables = decentralized(qTables, states, new_actions, alpha, r, gamma, new_states)
            states = new_states

    countNot0(qTables)

    pkl.dump(qTables[0], open('./QTables/qT1_Decentralized.p', 'wb'))
    pkl.dump(qTables[1], open('./QTables/qT2_Decentralized.p', 'wb'))
    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Decentralized.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Decentralized.csv')


def trainHysteretic():
    # generate Q-Table
    qTable1 = defaultdict(floatDD)
    qTable2 = defaultdict(floatDD)
    qTables = [qTable1, qTable2]
    trials = 5000
    for trial in range(trials):
        printProgressBar(trial, trials, prefix='Hysteretic: ')
        states = (-1.15, -3.2, 0, 0)
        for t in np.arange(0, 20, 0.03):
            new_actions = choose_action(states, actions, qTables, trial, trials=trials)

            theta1, v1 = getNextTheta1States(tau1=new_actions[0], tau2=new_actions[1], theta1=states[0],
                                             theta2=states[1], v1=states[2], t=0.03)
            theta2, v2 = getNextTheta2States(theta2=states[1], t=samplingTime, tau2=new_actions[1], v2=states[3])

            # speed saturation
            if v1 > 2 * np.pi: v1 = 2 * np.pi
            if v2 > 2 * np.pi: v2 = 2 * np.pi
            if v1 < -2 * np.pi: v1 = -2 * np.pi
            if v2 < -2 * np.pi: v2 = -2 * np.pi

            r1 = reward(states[0], states[2])
            r2 = reward(states[1], states[3])
            r = [r1, r2]

            new_states = (theta1, theta2, v1, v2)
            new_states = check_states(new_states)

            qTables = hysteretic(qTables, states, new_actions, alpha, beta, r, gamma, new_states)
            states = new_states

    countNot0(qTables)

    pkl.dump(qTables[0], open('./QTables/qT1_Hysteretic.p', 'wb'))
    pkl.dump(qTables[1], open('./QTables/qT2_Hysteretic.p', 'wb'))
    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Hysteretic.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Hysteretic.csv')


def trainCentralized():
    # generate Q-Table
    qTable = defaultdict(floatDD)
    trials = 5000
    for trial in range(trials):
        states = (-1.15, -3.2, 0, 0)
        printProgressBar(trial, trials, prefix='Centralized: ')
        for t in np.arange(0, 10, 0.03):
            new_actions = choose_action(states, actions, qTable, trial, centralized=True, trials=trials)
            # theta1, theta2, v1, v2 = getNexstates(np.array([[states[0]], [states[1]]]),
            #                                       np.array([[states[2]], [states[3]]]),
            #                                       np.array([[new_actions[0]], [new_actions[1]]]))
            theta1, v1 = getNextTheta1States(tau1=new_actions[0], tau2=new_actions[1], theta1=states[0],
                                             theta2=states[1], v1=states[2], t=0.03)
            theta2, v2 = getNextTheta2States(theta2=states[1], t=samplingTime, tau2=new_actions[1], v2=states[3])

            #speed saturation
            if v1 > 2 * np.pi: v1 = 2 * np.pi
            if v2 > 2 * np.pi: v2 = 2 * np.pi
            if v1 < -2 * np.pi: v1 = -2 * np.pi
            if v2 < -2 * np.pi: v2 = -2 * np.pi

            r = rewardCentralized([states[0],states[1]],[states[2],states[3]])

            new_states = (theta1, theta2, v1, v2)
            new_states = check_states(new_states)

            qTable = centralized(states, new_actions, r, gamma, alpha, qTable, new_states)
            states = new_states

    countNot0([qTable])

    pkl.dump(qTable, open('./QTables/qT_Centralized.p', 'wb'))
    pd.DataFrame.from_dict(qTable, orient='index').to_csv('./QTables/qT_Centralized.csv')


# trainDistributed()
# trainDecentralized()
# trainHysteretic()
trainCentralized()
