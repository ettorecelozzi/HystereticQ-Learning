from BallBalancing.Environment import *
import matplotlib.pyplot as plt
import pickle as pkl


def testBallBalancing(qTables, algorithm, centralized=False):
    """
    Play the game! Verify how much the robots have learnt.
    :param qTables: trained Q-Tables
    :param algorithm: Q-Algorithm used
    """
    outputSpace, outputSpeed = [], []
    states = (0.49, 0.06)  # initial states
    outputSpace.append(states[0])
    outputSpeed.append(states[1])
    for i in range(30):  # infinite game loop
        actions = choose_action(states, None, qTables, None, centralized)
        x, v = getNextStates(h1=actions[0], h2=actions[1], v=states[1], t=0.03, x_0=states[0], v_0=states[1])
        outputSpace.append(x)
        outputSpeed.append(v)
        if np.abs(x) > 1: break  # the ball has fallen
        new_states = (np.round(x, decimals=2), np.round(v, decimals=2))
        new_states = check_states(new_states)  # check if the states have a match in the discrete grid
        states = new_states
    plt.plot(outputSpace, '-', label="Space")
    plt.plot(outputSpeed, '-', label="Speed")
    plt.legend()
    plt.title(algorithm)
    plt.savefig('./Plots/' + algorithm + '_test.png')
    plt.clf()


for algorithm in ['Distributed', 'Decentralized', 'Hysteretic']:
    with open('./QTables/qT1_' + algorithm + '.p', 'rb') as file:
        qTable1 = pkl.load(file)
    with open('./QTables/qT2_' + algorithm + '.p', 'rb') as file:
        qTable2 = pkl.load(file)
    qTables = [qTable1, qTable2]
    testBallBalancing(qTables, algorithm)

algorithm = 'Centralized'
with open('./QTables/qT_' + algorithm + '.p', 'rb') as file:
    qTable = pkl.load(file)
testBallBalancing(qTable, algorithm, centralized=True)
