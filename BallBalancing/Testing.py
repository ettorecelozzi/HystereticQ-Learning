from BallBalancing.Environment import *
import matplotlib.pyplot as plt
import pickle as pkl


def testBallBalancing(qTables):
    """
    Play the game! Verify how much the robots have learnt.
    :param qTables: trained Q-Tables
    """
    # allSpaces = np.round(list(np.linspace(-1, 1, 100)), decimals=2)
    # allSpeeds = np.round(list(np.linspace(-3, 3, 50)), decimals=2)
    outputSpace, outputSpeed = [], []
    states = (0.49, 0.06)  # initial states
    outputSpace.append(states[0])
    outputSpeed.append(states[1])
    for i in range(30):  # infinite game loop
        # states = (np.random.choice(allSpaces), np.random.choice(allSpeeds))  # random initial state
        actions = choose_action(states, None, qTables, None)
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
    plt.show()


algorithm = 'hysteretic'
qTable1 = pkl.load(open('./QTables/qT1_' + algorithm + '.p', 'rb'))
qTable2 = pkl.load(open('./QTables/qT2_' + algorithm + '.p', 'rb'))
qTables = [qTable1, qTable2]
testBallBalancing(qTables)
