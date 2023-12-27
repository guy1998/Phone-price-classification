from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np


# function to carry out the decimal scaling process
def decimal_scaling(data, numerical_features):
    scaled_data = data.copy()
    for feature in data.columns:
        # condition added so that the non-numerical values stay as they are
        if feature in numerical_features:
            magnitude = 10 ** (np.ceil(np.log10(np.abs(data[feature]).max())))
            scaled_data[feature] = data[feature] / magnitude
    return scaled_data


# function to do the normalization according to min-max technique
def min_max_normalizer(data, features_to_normalize):
    min_max_scaler = MinMaxScaler()
    data_minmax_scaled = pd.DataFrame(data=min_max_scaler.fit_transform(data[features_to_normalize]),
                                      columns=features_to_normalize)
    return data_minmax_scaled


# function to do the z-score normalization
def z_score_normalizer(data, features_to_normalize):
    standard_scaler = StandardScaler()
    standard_normalized_data = pd.DataFrame(data=standard_scaler.fit_transform(data[features_to_normalize]),
                                            columns=features_to_normalize)
    return standard_normalized_data


def data_loader(path):
    data = pd.read_csv(path)
    return data

