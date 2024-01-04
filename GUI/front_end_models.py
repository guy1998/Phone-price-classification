from GUI.proxy import *

front_end_models = {
    "Decision tree": apply_decision_tree,
    "Random forest": apply_random_forest,
    "Support vector machine": apply_svm,
    "Naive bayes": apply_naive_bayes,
    "Logistic regression": apply_logistic_regression,
    "Multi-layer perceptron": apply_mlp,
    "Log. regression + FNN": apply_hybrid_log,
    "Random forest + FNN": apply_hybrid_random_forest
}


def get_model_function_by_name(model_name):
    return front_end_models[model_name]