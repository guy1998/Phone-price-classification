import pandas as pd  # Create arrays and lists
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split  # Used to do the splitting of the dataset
from sklearn.metrics import accuracy_score  # Performance metrics
from Data_Manipulation.normalization import data_loader


def apply_naive_bayes_fe_resolution_dataset(normalization_type, train_split):
    x = data_loader("Datasets/fe_resolution_" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_naive_bayes_fe_screen_size_dataset(normalization_type, train_split):
    x = data_loader("Datasets/fe_screen_size_" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_naive_bayes_normal_dataset(normalization_type, train_split):
    x = data_loader("Datasets/" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_naive_bayes_raw_dataset(train_split):
    x = data_loader("../train.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    print(x_train)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def naive_bayes(dataset="raw", normalization="raw", train_split=0.8):
    if dataset == "normal":
        apply_naive_bayes_normal_dataset(normalization, train_split)
    elif dataset == "fe_resolution":
        apply_naive_bayes_fe_resolution_dataset(normalization, train_split)
    elif dataset == "fe_screen_size":
        apply_naive_bayes_fe_screen_size_dataset(normalization, train_split)
    elif dataset == "raw":
        apply_naive_bayes_raw_dataset(train_split)
    else:
        raise Exception("No such dataset!")

