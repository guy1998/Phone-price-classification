
continuous_features_dictionary = {
    'Battery power(mAh)': 'battery_power',
    'Clock speed': 'clock_speed',
    'Front camera(Mpx)': 'fc',
    'Memory': 'int_memory',
    'Depth(cm)': 'm_dep',
    'Weight': 'mobile_wt',
    'Number of cores': 'n_cores',
    'Primary camera(Mpx)': 'pc',
    'Pixel width': 'px_width',
    'Pixel height': 'px_height',
    'RAM(Mb)': 'ram',
    'Screen height': 'sc_h',
    'Screen width': 'sc_w',
    'Talk time': 'talk_time'
}

categorical_features_dictionary = {
    'Bluetooth': 'blue',
    'Dual sim': 'dual_sim',
    '4G': 'four_g',
    '3G': 'three_g',
    'Touch screen': 'touch_screen',
    'Wifi': 'wifi'
}

user_input = {
    'battery_power': 0,
    'blue': 0,
    'clock_speed': 0,
    'dual_sim': 0,
    'fc': 0,
    'four_g': 0,
    'int_memory': 0,
    'm_dep': 0,
    'mobile_w': 0,
    'n_cores': 0,
    'pc': 0,
    'px_height': 0,
    'px_width': 0,
    'ram': 0,
    'sc_h': 0,
    'sc_w': 0,
    'talk_time': 0,
    'three_g': 0,
    'touch_screen': 0,
    'wifi': 0
}


def get_continuous_features():
    return continuous_features_dictionary.keys()


def get_categorical_features():
    return categorical_features_dictionary.keys()


def reset_user_input():
    global user_input
    user_input = {
        'battery_power': 0,
        'blue': 0,
        'clock_speed': 0,
        'dual_sim': 0,
        'fc': 0,
        'four_g': 0,
        'int_memory': 0,
        'm_dep': 0,
        'mobile_w': 0,
        'n_cores': 0,
        'pc': 0,
        'px_height': 0,
        'px_width': 0,
        'ram': 0,
        'sc_h': 0,
        'sc_w': 0,
        'talk_time': 0,
        'three_g': 0,
        'touch_screen': 0,
        'wifi': 0
    }


def set_user_input(key, value):
    if key in categorical_features_dictionary:
        user_input[categorical_features_dictionary[key]] = value
    else:
        user_input[continuous_features_dictionary[key]] = value


def get_user_input():
    return user_input
