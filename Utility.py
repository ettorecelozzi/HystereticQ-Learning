import numpy as np


def check_states(states):
    """
    Verify if the new states belong to the discrete grid
    :param states: float states
    :return: states in the discrete grid
    """
    positions = np.round(list(np.linspace(-1, 1, 100)), decimals=2)
    velocities = np.round(list(np.linspace(-3, 3, 50)), decimals=2)
    position_space = 1 / 50
    velocity_space = 3 / 25
    position_index = np.abs(int(np.round(states[0] / position_space, decimals=0)))
    velocity_index = np.abs(int(np.round(states[1] / velocity_space, decimals=0)))

    if states[0] > 0:
        position_index += 50
    if states[1] > 0:
        velocity_index += 25

    if position_index == 100:
        possible_positions = positions[position_index - 2:]
    elif position_index == 0:
        possible_positions = positions[:position_index + 2]
    elif position_index == 99:
        possible_positions = positions[position_index - 1:position_index + 1]
    else:
        possible_positions = positions[position_index - 1:position_index + 2]

    if velocity_index == 50:
        possible_velocities = velocities[velocity_index - 2:]
    elif velocity_index == 0:
        possible_velocities = velocities[:velocity_index + 2]
    elif velocity_index == 49:
        possible_velocities = velocities[velocity_index - 1:velocity_index + 1]
    else:
        possible_velocities = velocities[velocity_index - 1:velocity_index + 2]

    new_states = (min(possible_positions, key=lambda x: abs(x - states[0])),
                  min(possible_velocities, key=lambda x: abs(x - states[1])))
    return new_states


def check_actions(new_actions, actions):
    """
    Verify if the new action have a match in the discrete grid
    :param new_actions: tuple of action to verify
    :param actions: list of all discrete action available
    :return: discrete actions if needed
    """
    discrete_actions = [0.0] * len(new_actions)
    discrete = False
    for a in range(len(new_actions)):
        if new_actions[a] not in actions:
            discrete_actions[a] = min(actions, key=lambda x: abs(x - new_actions[a]))
            discrete = True
    return discrete_actions if discrete else new_actions


def countNot0(qTables):
    """
    Count the number of element that are not 0 in the QTables
    :param qTables: list of the qTables
    :return: number of elment != 0
    """
    counter = [0] * len(qTables)
    qTable_index = 0
    for q in qTables:
        for state in q:
            for a in q[state]:
                if q[state][a] != 0.0:
                    counter[qTable_index] += 1
        qTable_index += 1
    print(counter)
