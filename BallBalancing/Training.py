from BallBalancing.QLearningAlgorithms import *
import pandas as pd
import matplotlib.pyplot as plt
from BallBalancing.Environment import *
import pickle as pkl

# parameters
alpha = 0.9
beta = 0.1
gamma = 0.9
samplingTime = 0.03
decimals = 3
trials = 5000
actions = np.round(np.linspace(-1, 1, 15), decimals=decimals)



def trainDecentralized():
    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]

    output20 = []
    for i in range(5):
        output = []
        for trial in range(trials):
            printProgressBar(trial, trials, prefix='Iteration: ' + str(i))
            states = (0.495, 0.061)  # initial states
            rewardSum = 0
            for t in np.arange(0, 20, samplingTime):
                new_actions = choose_action(states, actions, qTables, trial, numOfEps=40, trials=trials)

                # dynamic computed inside
                x, v = getNextStates(h1=new_actions[0], h2=new_actions[1], v=states[1], t=samplingTime, x_0=states[0],
                                     v_0=states[1])
                # saturation of the speed
                if v > 3: v = 3
                if v < -3: v = -3

                if np.abs(x) > 1: break  # the ball has fallen

                r = reward(states[0], states[1])
                rewardSum += r

                new_states = (np.round(x, decimals=decimals), np.round(v, decimals=decimals))
                new_states = check_states(new_states)  # check if the states have a match in the discrete grid

                qTables = decentralized(qTables, states, new_actions, alpha, r, gamma, new_states)
                states = new_states
            output.append(rewardSum)
        output20.append(output)

    mean_output = np.mean(output20, axis=0)
    plt.plot(list(range(trials)), mean_output, '-')
    plt.title('Decentralized')
    plt.savefig('./Plots/Decentralized.png')
    plt.clf()

    countNot0(qTables)

    pkl.dump(qTables[0], open('./QTables/qT1_Decentralized.p', 'wb'))
    pkl.dump(qTables[1], open('./QTables/qT2_Decentralized.p', 'wb'))
    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Decentralized.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Decentralized.csv')


def trainHysteretic():
    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]

    output20 = []
    for i in range(5):
        output = []
        for trial in range(trials):
            printProgressBar(trial, trials, prefix='Iteration: ' + str(i))
            states = (0.495, 0.061)  # initial states
            rewardSum = 0

            for t in np.arange(0, 20, samplingTime):

                new_actions = choose_action(states, actions, qTables, trial, numOfEps=40, trials=trials)

                # dynamic computed inside
                x, v = getNextStates(h1=new_actions[0], h2=new_actions[1], v=states[1], t=samplingTime, x_0=states[0],
                                     v_0=states[1])

                # saturation of the speed
                if v > 3: v = 3
                if v < -3: v = -3

                if np.abs(x) > 1: break

                r = reward(states[0], states[1])
                rewardSum = rewardSum + r

                new_states = (np.round(x, decimals=decimals), np.round(v, decimals=decimals))
                new_states = check_states(new_states)  # check if the states have a match in the discrete grid

                qTables = hysteretic(qTables, states, new_actions, alpha, beta, r, gamma, new_states)
                states = new_states

            output.append(rewardSum)
        output20.append(output)
    mean_output = np.mean(output20, axis=0)
    plt.plot(list(range(trials)), mean_output, '-')
    plt.title('Hysteretic')
    plt.savefig('./Plots/Hysteretic.png')
    plt.clf()

    countNot0(qTables)

    pkl.dump(qTables[0], open('./QTables/qT1_Hysteretic.p', 'wb'))
    pkl.dump(qTables[1], open('./QTables/qT2_Hysteretic.p', 'wb'))
    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Hysteretic.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Hysteretic.csv')


def trainCentralized():
    # generate Q-Table
    qTable = generateQTable(centralized=True)

    output20 = []
    for i in range(5):
        output = []
        for trial in range(trials):
            printProgressBar(trial, trials, prefix='Iteration: ' + str(i))
            states = (0.495, 0.061)  # initial states
            rewardSum = 0
            for t in np.arange(0, 20, samplingTime):
                new_actions = choose_action(states, actions, qTable, trial, centralized=True, numOfEps=40,trials=trials)

                # dynamic computed inside
                x, v = getNextStates(h1=new_actions[0], h2=new_actions[1], v=states[1], t=samplingTime, x_0=states[0],
                                     v_0=states[1])
                # saturation of the speed
                if v > 3: v = 3
                if v < -3: v = -3

                if np.abs(x) > 1: break  # the ball has fallen

                r = reward(states[0], states[1])
                rewardSum += r

                new_states = (np.round(x, decimals=decimals), np.round(v, decimals=decimals))
                new_states = check_states(new_states)  # check if the states have a match in the discrete grid

                qTable = centralized(states, new_actions, r, gamma, alpha, qTable, new_states)
                states = new_states

            output.append(rewardSum)
        output20.append(output)

    mean_output = np.mean(output20, axis=0)
    plt.plot(list(range(trials)), mean_output, '-')
    plt.title('Centralized')
    plt.savefig('./Plots/Centralized.png')
    plt.clf()

    countNot0([qTable])

    pkl.dump(qTable, open('./QTables/qT_Centralized.p', 'wb'))
    pd.DataFrame.from_dict(qTable, orient='index').to_csv('./QTables/qT_Centralized.csv')


trainDecentralized()
trainHysteretic()
trainCentralized()
