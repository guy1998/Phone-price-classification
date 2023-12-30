from Algorithms.feed_forward_neural_network import mlp_model
from Algorithms.ensemble_models import ensemble_model, ensemble_model_logistic_regression
from Algorithms.decision_tree import decision_tree

if __name__ == "__main__":
    mlp_model("fe_screen_size", "z_score", n_hidden=20, activation="tanh")
    # ensemble_model_logistic_regression("fe_screen_size", "z_score")
