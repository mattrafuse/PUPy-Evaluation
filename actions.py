def enumerate_actions(old_state, new_state):
    actions = device_auth(old_state, new_state)
    actions = actions + device_theft(old_state, new_state)
    actions = actions + device_loss(old_state, new_state)

    return actions


def device_auth(old_state, new_state):
    actions = []

    # if old_state[1] - old_state[2] >= -0.2 and new_state[1] - new_state[2] < -0.2:
    #     actions.append((0, 'Start Auth High Alert'))
    # elif old_state[1] - old_state[2] < -0.2 and new_state[1] - new_state[2] >= -0.2:
    #     actions.append((0, 'End Auth High Alert'))

    if new_state[1] - new_state[2] < -.4:
        actions.append((0, 2))
    elif new_state[1] - new_state[2] < .1:
        actions.append((0, 1))
    elif new_state[1] - new_state[2] >= .1:
        actions.append((0, 0))

    return actions


def device_theft(old_state, new_state):
    actions = []

    if new_state[2] > .5 and new_state[3] < .9:
        actions.append((1, 1))
    elif new_state[2] <= .1 or new_state[3] >= 1:
        actions.append((1, 0))

    return actions


def device_loss(old_state, new_state):
    actions = []

    if new_state[3] > .85:
        actions.append((2, 0))
    elif new_state[3] <= .75:
        actions.append((2, 1))

    # if old_state[3] >= .3 and new_state[3] < .3:
    #     if old_state[1] < .5:
    #         actions.append((2, 'Send Lost Device Notification'))

    return actions
