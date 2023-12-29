from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Data_Manipulation.normalization import data_loader
import pandas as pd


def apply_mlp_fe_resolution(normalization_type, train_size, n_hidden, optimizer, activation):
    x = data_loader("Datasets/fe_resolution_" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)
    mlp = MLPClassifier(verbose=True, hidden_layer_sizes=(n_hidden,), solver=optimizer, activation=activation,
                        max_iter=10000,
                        random_state=42)
    mlp.fit(x_train, y_train)
    y_predict = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_mlp_fe_screen_size(normalization_type, train_size, n_hidden, optimizer, activation):
    x = data_loader("Datasets/fe_screen_size_" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)
    mlp = MLPClassifier(verbose=True, hidden_layer_sizes=(n_hidden,), solver=optimizer, activation=activation,
                        max_iter=10000,
                        random_state=42)
    mlp.fit(x_train, y_train)
    y_predict = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")

def apply_mlp_normal(normalization_type, train_size, n_hidden, optimizer, activation):
    x = data_loader("Datasets/" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)
    mlp = MLPClassifier(verbose=True, hidden_layer_sizes=(n_hidden,), solver=optimizer, activation=activation,
                        max_iter=10000,
                        random_state=42)
    mlp.fit(x_train, y_train)
    y_predict = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")

def apply_mlp_raw(train_size, n_hidden, optimizer, activation):
    x = data_loader("train.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)
    mlp = MLPClassifier(verbose=True, hidden_layer_sizes=(n_hidden,), solver=optimizer, activation=activation, max_iter=10000,
                        random_state=42)
    mlp.fit(x_train, y_train)
    y_predict = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def mlp_model(dataset="raw", normalization_type="raw", train_size=0.8, n_hidden=18, optimizer='adam', activation='softmax'):
    if dataset == "fe_resolution":
        apply_mlp_fe_resolution(normalization_type, train_size, n_hidden, optimizer, activation)
    elif dataset == "fe_screen_size":
        apply_mlp_fe_screen_size(normalization_type, train_size, n_hidden, optimizer, activation)
    elif dataset == "normal":
        apply_mlp_normal(normalization_type, train_size, n_hidden, optimizer, activation)
    elif dataset == "raw":
        apply_mlp_raw(train_size, n_hidden, optimizer, activation)