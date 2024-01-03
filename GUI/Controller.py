STATE = 1
MODEL = 'NONE'
changed = False


def change_state(new_state):
    print("new state:" + str(new_state))
    global STATE
    STATE = new_state


def get_state():
    return STATE


def change_model(new_model):
    global MODEL
    MODEL = new_model


def get_model():
    return MODEL


def get_changed():
    return changed


def change_true():
    global changed
    changed = True


def change_false():
    global changed
    changed = False

