from Algorithms.algorithms_ui import *
import pandas as pd
from GUI.input_tracker import get_user_input
from Data_Manipulation.normalization import decimal_scaling
from GUI.results import create_result_prompt


def apply_mlp():
    names_of_categorical_features = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
    model, accuracy = mlp_ui()
    df = pd.DataFrame(get_user_input(), index=[0])
    numerical_features = [x for x in list(df.iloc[:0, :]) if x not in names_of_categorical_features]
    df = decimal_scaling(df, numerical_features)
    prediction = model.predict(df)
    create_result_prompt(prediction, accuracy)


def apply_decision_tree():
    pass


def apply_svm():
    pass


def apply_logistic_regression():
    pass


def apply_naive_bayes():
    pass


def apply_random_forest():
    pass


def apply_hybrid_log():
    pass


def apply_hybrid_random_forest():
    pass