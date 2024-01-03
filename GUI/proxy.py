from Algorithms.algorithms_ui import *
import pandas as pd
from GUI.input_tracker import get_user_input


def apply_mlp():
    model, accuracy = mlp_ui()
    df = pd.DataFrame(get_user_input())
    prediction = model.predict(df)
    print(prediction)


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
