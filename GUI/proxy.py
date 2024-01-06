from Algorithms.algorithms_ui import *
import pandas as pd
from GUI.input_tracker import get_user_input
from Data_Manipulation.normalization import decimal_scaling, z_score_normalizer
from GUI.results import create_result_prompt
from Data_Manipulation.feature_engineering import create_dataset_with_screen_size
import numpy as np


names_of_categorical_features = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]


def apply_mlp():
    model, accuracy = mlp_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    numerical_features = [x for x in list(df.iloc[:0, :-1]) if x not in names_of_categorical_features]
    df = decimal_scaling(df, numerical_features)
    prediction = model.predict(df)
    create_result_prompt(prediction, accuracy)


def apply_decision_tree():
    model, accuracy = decision_tree_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    df = create_dataset_with_screen_size(df)
    numerical_features = [x for x in list(df.iloc[:0, :-1]) if x not in names_of_categorical_features]
    df = decimal_scaling(df, numerical_features)
    prediction = model.predict(df)
    create_result_prompt(prediction, accuracy)


def apply_svm():
    model, accuracy = svm_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    prediction = model.predict(df)
    create_result_prompt(prediction, accuracy)


def apply_logistic_regression():
    model, accuracy = logistic_regression_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    df = create_dataset_with_screen_size(df)
    numerical_features = [x for x in list(df.iloc[:0, :-1]) if x not in names_of_categorical_features]
    df = pd.concat([z_score_normalizer(df, numerical_features),
                                    df.loc[:, names_of_categorical_features]], axis=1)
    prediction = model.predict(df)
    create_result_prompt(prediction, accuracy)


def apply_naive_bayes():
    model, accuracy = naive_bayes_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    df = create_dataset_with_screen_size(df)
    numerical_features = [x for x in list(df.iloc[:0, :-1]) if x not in names_of_categorical_features]
    df = decimal_scaling(df, numerical_features)
    prediction = model.predict(df)
    create_result_prompt(prediction, accuracy)


def apply_random_forest():
    model, accuracy = random_forest_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    df = create_dataset_with_screen_size(df)
    numerical_features = [x for x in list(df.iloc[:0, :-1]) if x not in names_of_categorical_features]
    df = pd.concat([z_score_normalizer(df, numerical_features),
                    df.loc[:, names_of_categorical_features]], axis=1)
    prediction = model.predict(df)
    create_result_prompt(prediction, accuracy)


def apply_hybrid_log():
    log_model, model, accuracy = ensemble_log_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    df = create_dataset_with_screen_size(df)
    numerical_features = [x for x in list(df.iloc[:0, :-1]) if x not in names_of_categorical_features]
    df = pd.concat([z_score_normalizer(df, numerical_features),
                    df.loc[:, names_of_categorical_features]], axis=1)
    log_prediction = log_model.predict_proba(df)
    nn_prediction = model.predict(df)
    combined_prediction_proba = log_prediction + nn_prediction
    combined_prediction = np.argmax(combined_prediction_proba, axis=1)
    create_result_prompt(combined_prediction, accuracy)


def apply_hybrid_random_forest():
    random_model, model, accuracy = ensemble_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    df = create_dataset_with_screen_size(df)
    numerical_features = [x for x in list(df.iloc[:0, :-1]) if x not in names_of_categorical_features]
    df = pd.concat([z_score_normalizer(df, numerical_features),
                    df.loc[:, names_of_categorical_features]], axis=1)
    random_prediction = random_model.predict_proba(df)
    nn_prediction = model.predict(df)
    combined_prediction_proba = random_prediction + nn_prediction
    combined_prediction = np.argmax(combined_prediction_proba, axis=1)
    create_result_prompt(combined_prediction, accuracy)
