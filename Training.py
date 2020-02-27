from QLearningAlgorithms import *
from Utility import *
import pandas as pd


def trainDistributed():
    # parameters
    alpha = 0.9

    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]
    actions = np.round(np.linspace(-1, 1, 15), decimals=1)
    output = []
    for trial in range(5000):
        states = (0.5, 0.1)  # initial states
        rewardSum = 0
        for t in np.arange(0, 20, 0.03):
            new_actions = choose_action(states, actions, qTables)

            # dynamic computed inside
            x, v = getNextStates(h1=new_actions[0], h2=new_actions[1], v=states[1], t=t, x_0=states[0], v_0=states[1])
            if np.abs(x) > 1 or np.abs(v) > 3: break  # the ball has fallen

            r = reward(x, v)
            rewardSum = rewardSum + r
            states = (np.round(x, decimals=1), np.round(v, decimals=1))
            states = check_states(states, qTables)  # check if the states have a match in the discrete grid

            new_actions = np.round(new_actions, decimals=1)
            new_actions = check_actions(new_actions, actions)

            qTables = distributed(qTables, r, states, new_actions, alpha)
        output.append(rewardSum)
        # if trial == 0:
        #     output.append(rewardSum)
        # else:
        #     output.append(output[trial-1]+rewardSum)
    plt.scatter(list(range(5000)), output)
    plt.show()

    countNot0(qTables)

    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Distributed.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Distributed.csv')


def trainDecentralized():
    # parameters
    alpha = 0.9
    gamma = 0.9

    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]
    actions = np.round(np.linspace(-1, 1, 15), decimals=1)
    output = []
    for trial in range(5000):
        states = (0.5, 0.1)  # initial states
        rewardSum = 0
        for t in np.arange(0, 20, 0.03):
            new_actions = choose_action(states, actions, qTables)

            # dynamic computed inside
            x, v = getNextStates(h1=new_actions[0], h2=new_actions[1], v=states[1], t=t, x_0=states[0], v_0=states[1])
            if np.abs(x) > 1 or np.abs(v) > 3: break  # the ball has fallen

            r = reward(x, v)
            rewardSum = rewardSum + r
            new_states = (np.round(x, decimals=1), np.round(v, decimals=1))
            new_states = check_states(new_states, qTables)  # check if the states have a match in the discrete grid

            new_actions = np.round(new_actions, decimals=1)  # check if the actions have a match in the discrete grid
            new_actions = check_actions(new_actions, actions)

            qTables = decentralized(qTables, states, new_actions, alpha, r, gamma, new_states)
            states = new_states
        output.append(rewardSum)
        # if trial == 0:
        #     output.append(rewardSum)
        # else:
        #     output.append(output[trial-1]+rewardSum)
    plt.scatter(list(range(5000)), output)
    plt.show()
    print(qTables[0])
    countNot0(qTables)

    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Decentralized.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Decentralized.csv')


def trainHysteretic():
    # parameters
    alpha = 0.9
    beta = 0.1
    gamma = 0.9

    # generate Q-Table
    qTable1 = generateQTable()
    qTable2 = generateQTable()
    qTables = [qTable1, qTable2]
    actions = np.round(np.linspace(-1, 1, 15), decimals=1)
    output = []
    for trial in range(5000):
        states = (0.5, 0.1)  # initial states
        rewardSum = 0
        for t in np.arange(0, 20, 0.03):
            new_actions = choose_action(states, actions, qTables)

            # dynamic computed inside
            x, v = getNextStates(h1=new_actions[0], h2=new_actions[1], v=states[1], t=t, x_0=states[0], v_0=states[1])
            if np.abs(x) > 1 or np.abs(v) > 3: break  # the ball has fallen

            r = reward(states[0], states[1])
            rewardSum = rewardSum + r
            new_states = (np.round(x, decimals=1), np.round(v, decimals=1))
            new_states = check_states(new_states, qTables)  # check if the states have a match in the discrete grid

            new_actions = np.round(new_actions, decimals=1)
            new_actions = check_actions(new_actions, actions)

            qTables = hysteretic(qTables, states, new_actions, alpha, beta, r, gamma, new_states)
            states = new_states
        output.append(rewardSum)
        # if trial == 0:
        #     output.append(rewardSum)
        # else:
        #     output.append(output[trial-1]+rewardSum)
    plt.scatter(list(range(5000)), output)
    plt.show()
    print(len(output))
    print(qTables[0])
    countNot0(qTables)

    pd.DataFrame.from_dict(qTables[0], orient='index').to_csv('./QTables/qT1_Hysteretic.csv')
    pd.DataFrame.from_dict(qTables[1], orient='index').to_csv('./QTables/qT2_Hysteretic.csv')

print('Distributed case')
trainDistributed()
print('Decentralized case')
trainDecentralized()
print('Hysteteric case')
trainHysteretic()