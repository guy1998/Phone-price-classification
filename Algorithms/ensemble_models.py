import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from Data_Manipulation.normalization import data_loader


def data_splitter(path, train_size):
    df = data_loader(path)
    y = df['price_range']
    x = df.iloc[:, :-1]
    if path != "train.csv":
        x = df.iloc[:, 1:-1]
    return train_test_split(x, y, train_size=train_size, random_state=42)


def ensemble(input_size, data_tuple):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(data_tuple[0], data_tuple[2])
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(data_tuple[2].to_numpy().reshape(-1, 1))
    model = Sequential()
    model.add(Dense(input_size+1, input_dim=input_size, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data_tuple[0], y_train_encoded, epochs=50, batch_size=8, verbose=0)
    y_pred_ensemble = np.argmax(model.predict(data_tuple[1]) + clf.predict_proba(data_tuple[1]), axis=1)
    accuracy_ensemble = accuracy_score(data_tuple[3], y_pred_ensemble)
    print(f'Accuracy of Ensemble Model: {accuracy_ensemble:.2f}')


def apply_ensemble_fe_resolution(normalization_type, train_size):
    data_tuple = data_splitter("Datasets/fe_resolution_" + normalization_type + "_dataset.csv"
                                                     , train_size)
    ensemble(17, data_tuple)


def apply_ensemble_fe_screen_size(normalization_type, train_size):
    data_tuple = data_splitter("Datasets/fe_screen_size_" + normalization_type + "_dataset.csv"
                               , train_size)
    ensemble(19, data_tuple)


def apply_ensemble_normal(normalization_type, train_size):
    data_tuple = data_splitter("Datasets/" + normalization_type + "_dataset.csv"
                               , train_size)
    ensemble(20, data_tuple)


def apply_ensemble_raw_data(train_size):
    data_tuple = data_splitter("train.csv", train_size)
    ensemble(20, data_tuple)


def ensemble_model(dataset="raw", normalization_type="raw", train_size=0.8):
    if dataset == "fe_resolution":
        apply_ensemble_fe_resolution(normalization_type, train_size)
    elif dataset == "fe_screen_size":
        apply_ensemble_fe_screen_size(normalization_type, train_size)
    elif dataset == "normal":
        apply_ensemble_normal(normalization_type, train_size)
    elif dataset == "raw":
        apply_ensemble_raw_data(train_size)

