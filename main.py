from Algorithms.logistic_regression import logistic_regression
from Algorithms.svm import svm_model
from Algorithms.naive_bayes import naive_bayes
from Algorithms.random_forest import random_forest

if __name__ == "__main__":
    logistic_regression("fe_resolution", "z_score", max_iter=10000)
    svm_model("normal", "z_score")
    naive_bayes('fe_resolution', 'min_max')
    random_forest("normal", "min_max", n_estimators=1000)
