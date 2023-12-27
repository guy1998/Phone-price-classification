def create_dataset_with_resolution(data):
    # very small value added to avoid division with near zero numbers
    # does not affect the result that much
    epsilon = 0.000000001
    resolution = (data['px_width'] / (data['sc_w'] + epsilon)) * (data['px_height'] / (data['sc_h'] + epsilon))
    data.insert(3, "resolution", resolution)
    features_to_drop = ['px_width', 'px_height', 'sc_w', 'sc_h']
    data = data.drop(columns=features_to_drop)
    return data


def create_dataset_with_screen_size(data):
    features_to_drop = ['sc_w', 'sc_h']
    data.insert(3, "screen_size", (data['sc_w'] * data['sc_h']))
    data = data.drop(columns=features_to_drop)
    return data


