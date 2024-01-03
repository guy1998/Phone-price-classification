import pandas as pd  # Create arrays and lists
from sklearn.tree import DecisionTreeClassifier  # Used to import the Decision Tree method, already coded there
from sklearn.model_selection import train_test_split  # Used to do the splitting of the dataset
from sklearn.metrics import accuracy_score  # Performance metrics
from Data_Manipulation.normalization import data_loader


def apply_decision_tree_fe_resolution_dataset(normalization_type, train_split, criterion):
    x = data_loader("Datasets/fe_resolution_" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_decision_tree_fe_screen_size_dataset(normalization_type, train_split, criterion):
    x = data_loader("Datasets/fe_screen_size_" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_decision_tree_normal_dataset(normalization_type, train_split, criterion):
    x = data_loader("Datasets/" + normalization_type + "_dataset.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def apply_decision_tree_raw_dataset(train_split, criterion):
    x = data_loader("train.csv")
    y = pd.Series(x['price_range'])
    x = x.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_split), random_state=7)
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {accuracy}")


def decision_tree(dataset="raw", normalization="raw", train_split=0.8, criterion='gini'):
    if dataset == "normal":
        apply_decision_tree_normal_dataset(normalization, train_split, criterion)
    elif dataset == "fe_resolution":
        apply_decision_tree_fe_resolution_dataset(normalization, train_split, criterion)
    elif dataset == "fe_screen_size":
        apply_decision_tree_fe_screen_size_dataset(normalization, train_split, criterion)
    elif dataset == "raw":
        apply_decision_tree_raw_dataset(train_split, criterion)
    else:
        raise Exception("No such dataset!")

