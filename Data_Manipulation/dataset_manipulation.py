from Data_Manipulation.normalization import data_loader, min_max_normalizer, decimal_scaling, z_score_normalizer
from feature_engineering import create_dataset_with_resolution, create_dataset_with_screen_size
import pandas as pd

# Normal dataset and the 2 datasets with feature engineering
mobile_phone_data = data_loader("../train.csv")
mobile_phone_data_FE_resolution = create_dataset_with_resolution(mobile_phone_data.copy())
mobile_phone_data_FE_screen_size = create_dataset_with_screen_size(mobile_phone_data.copy())

# here I have loaded the data and then have divided the attributes into numerical, categorical and target
# the 2 latter types will not be normalized
target_column = mobile_phone_data.iloc[:, [-1]]
names_of_categorical_features = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
categorical_features = mobile_phone_data.loc[:, names_of_categorical_features]
numerical_features = [x for x in list(mobile_phone_data.iloc[:0, :-1]) if x not in names_of_categorical_features]
numerical_features_FE_resolution = [x for x in list(mobile_phone_data_FE_resolution.iloc[:0, :-1])
                                    if x not in names_of_categorical_features]
numerical_features_FE_screen_size = [x for x in list(mobile_phone_data_FE_screen_size.iloc[:0, :-1])
                                     if x not in names_of_categorical_features]


# Datasets with min-max normalization
def create_datasets_with_min_max():
    normal_min_max_dataset = pd.concat([min_max_normalizer(mobile_phone_data, numerical_features),
                                        categorical_features, target_column], axis=1)
    fe_resolution_min_max_dataset = pd.concat([min_max_normalizer(mobile_phone_data_FE_resolution,
                                                                  numerical_features_FE_resolution),
                                               categorical_features,
                                               target_column], axis=1)
    fe_screen_size_min_max_dataset = pd.concat([min_max_normalizer(mobile_phone_data_FE_screen_size,
                                                                   numerical_features_FE_screen_size),
                                                categorical_features,
                                                target_column], axis=1)
    normal_min_max_dataset.to_csv("Datasets/min_max_dataset.csv")
    fe_screen_size_min_max_dataset.to_csv("Datasets/fe_screen_size_min_max_dataset.csv")
    fe_resolution_min_max_dataset.to_csv("Datasets/fe_resolution_min_max_dataset.csv")


# Datasets with decimal scaling normalization
def create_datasets_with_decimal():
    normal_decimal_dataset = decimal_scaling(mobile_phone_data, numerical_features)
    fe_resolution_decimal_dataset = decimal_scaling(mobile_phone_data_FE_resolution, numerical_features_FE_resolution)
    fe_screen_size_decimal_dataset = decimal_scaling(mobile_phone_data_FE_screen_size,
                                                     numerical_features_FE_screen_size)
    normal_decimal_dataset.to_csv("Datasets/decimal_dataset.csv")
    fe_resolution_decimal_dataset.to_csv("Datasets/fe_resolution_decimal_dataset.csv")
    fe_screen_size_decimal_dataset.to_csv("Datasets/fe_screen_size_decimal_dataset.csv")


# Datasets with z-score normalization
def create_datasets_with_z_score():
    normal_z_score_dataset = pd.concat([z_score_normalizer(mobile_phone_data, numerical_features),
                                        categorical_features, target_column], axis=1)
    fe_resolution_z_score_dataset = pd.concat([z_score_normalizer(mobile_phone_data_FE_resolution, numerical_features_FE_resolution),
                                               categorical_features, target_column], axis=1)
    fe_screen_size_z_score_dataset = pd.concat([z_score_normalizer(mobile_phone_data_FE_screen_size, numerical_features_FE_screen_size),
                                                categorical_features, target_column], axis=1)
    normal_z_score_dataset.to_csv("Datasets/z_score_dataset.csv")
    fe_resolution_z_score_dataset.to_csv("Datasets/fe_resolution_z_score_dataset.csv")
    fe_screen_size_z_score_dataset.to_csv("Datasets/fe_screen_size_z_score_dataset.csv")
