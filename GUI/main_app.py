import tkinter as tk
from tkinter import font
from GUI.Controller import change_state, change_true, change_model


def main_menu():

    def switching_states(value):
        change_state(2)
        change_true()
        change_model(value)

    frame = tk.Frame(width=650, height=440)

    title_label = tk.Label(frame, text="Phone price classificator",
                           font=font.Font(family="Verdana", size=25, weight="bold"))
    title_label.grid(row=0, column=0, columnspan=2)

    prompt_label = tk.Label(frame, text="Select the model you prefer:", font="Verdana")
    prompt_label.grid(row=1, column=0, columnspan=2)

    decision_tree_button = tk.Button(frame, text="Decision tree", width=30, height=2,
                                     command=lambda: switching_states("Decision tree"))
    decision_tree_button.grid(row=1+1, column=0, padx=(20, 20), pady=(20, 20))

    random_forest_button = tk.Button(frame, text="Random forest", width=30, height=2,
                                     command=lambda: switching_states("Random forest"))
    random_forest_button.grid(row=1+1, column=1, padx=(20, 20), pady=(20, 20))

    svm = tk.Button(frame, text="SVM", width=30, height=2,
                                     command=lambda: switching_states("Support vector machine"))
    svm.grid(row=2+1, column=0, padx=(20, 20), pady=(20, 20))

    naive_bayes = tk.Button(frame, text="Naive Bayes", width=30, height=2,
                                     command=lambda: switching_states("Naive bayes"))
    naive_bayes.grid(row=2+1, column=1, padx=(20, 20), pady=(20, 20))

    logistic_regression = tk.Button(frame, text="Logistic regression", width=30, height=2,
                                     command=lambda: switching_states("Logistic regression"))
    logistic_regression.grid(row=3+1, column=0, padx=(20, 20), pady=(20, 20))

    mlp = tk.Button(frame, text="Multi-layer perceptron", width=30, height=2,
                                     command=lambda: switching_states("Multi-layer perceptron"))
    mlp.grid(row=3+1, column=1, padx=(20, 20), pady=(20, 20))

    ensemble_1 = tk.Button(frame, text="Log. regression + FNN", width=30, height=2,
                                     command=lambda: switching_states("Log. regression + FNN"))
    ensemble_1.grid(row=4+1, column=0, padx=(20, 20), pady=(20, 20))

    ensemble_2 = tk.Button(frame, text="Random forest + FNN", width=30, height=2,
                                     command=lambda: switching_states("Random forest + FNN"))
    ensemble_2.grid(row=4+1, column=1, padx=(20, 20), pady=(20, 20))

    return frame
