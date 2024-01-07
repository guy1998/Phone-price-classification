from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from Data_Manipulation.normalization import data_loader
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd


def mlp_ui():
    x = data_loader("Datasets/decimal_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    mlp = MLPClassifier(verbose=False, hidden_layer_sizes=(5,), solver="adam", activation='identity',
                        max_iter=10000,
                        random_state=42, learning_rate='adaptive')
    mlp.fit(x_train, y_train)
    y_predict = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    return mlp, accuracy


def decision_tree_ui():
    x = data_loader("Datasets/fe_screen_size_decimal_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    return clf, accuracy


def svm_ui():
    x = data_loader("train.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    print(x_train)
    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    return clf, accuracy


def logistic_regression_ui():
    x = data_loader("Datasets/fe_screen_size_z_score_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    clf = LogisticRegression(max_iter=10000, solver="saga")
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    return clf, accuracy


def naive_bayes_ui():
    x = data_loader("Datasets/fe_screen_size_decimal_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=7)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    return clf, accuracy


def random_forest_ui():
    x = data_loader("Datasets/fe_screen_size_z_score_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=7)
    clf = RandomForestClassifier(n_estimators=500, criterion='gini')
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    return clf, accuracy


def ensemble_ui():
    df = data_loader("Datasets/fe_screen_size_z_score_dataset.csv")
    y = df['price_range']
    x = df.iloc[:, 1:-1]
    data_tuple = train_test_split(x, y, train_size=0.8, random_state=42)
    clf = RandomForestClassifier(random_state=42, criterion='gini', n_estimators=100)
    clf.fit(data_tuple[0], data_tuple[2])
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(data_tuple[2].to_numpy().reshape(-1, 1))
    model = Sequential()
    model.add(Dense(5, input_dim=19, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data_tuple[0], y_train_encoded, epochs=50, batch_size=8, verbose=0)
    y_pred_ensemble = np.argmax(model.predict(data_tuple[1]) + clf.predict_proba(data_tuple[1]), axis=1)
    accuracy_ensemble = accuracy_score(data_tuple[3], y_pred_ensemble)
    return clf, model, accuracy_ensemble


def ensemble_log_ui():
    df = data_loader("Datasets/fe_screen_size_z_score_dataset.csv")
    y = df['price_range']
    x = df.iloc[:, 1:-1]
    data_tuple = train_test_split(x, y, train_size=0.8, random_state=42)
    clf = LogisticRegression(random_state=42, solver='sag')
    clf.fit(data_tuple[0], data_tuple[2])
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(data_tuple[2].to_numpy().reshape(-1, 1))
    model = Sequential()
    model.add(Dense(5, input_dim=19, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data_tuple[0], y_train_encoded, epochs=50, batch_size=8, verbose=0)
    y_pred_ensemble = np.argmax(model.predict(data_tuple[1]) + clf.predict_proba(data_tuple[1]), axis=1)
    accuracy_ensemble = accuracy_score(data_tuple[3], y_pred_ensemble)
    return clf, model, accuracy_ensemble
