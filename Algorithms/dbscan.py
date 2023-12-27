from Data_Manipulation.normalization import data_loader
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def apply_dbscan_fe_resolution_dataset(normalization_type, eps, min_pts):
    x = data_loader("Datasets/fe_resolution_" + normalization_type + "_dataset.csv").iloc[:, :-1]
    db_scan = DBSCAN(eps=eps, min_samples=min_pts)
    predicted_labels = db_scan.fit_predict(x)
    silhouette = silhouette_score(x, predicted_labels)
    print(f"Silhouette: {silhouette}")


def apply_dbscan_normal_dataset(normalization_type, eps, min_pts):
    x = data_loader("Datasets/" + normalization_type + "_dataset.csv").iloc[:, :-1]
    db_scan = DBSCAN(eps=eps, min_samples=min_pts)
    predicted_labels = db_scan.fit_predict(x)
    silhouette = silhouette_score(x, predicted_labels)
    print(f"Silhouette: {silhouette}")


def apply_dbscan_fe_screen_size_dataset(normalization_type, eps, min_pts):
    x = data_loader("Datasets/fe_screen_size_" + normalization_type + "_dataset.csv").iloc[:, :-1]
    db_scan = DBSCAN(eps=eps, min_samples=min_pts)
    predicted_labels = db_scan.fit_predict(x)
    silhouette = silhouette_score(x, predicted_labels)
    print(f"Silhouette: {silhouette}")


def apply_dbscan_raw_dataset(eps, min_pts):
    x = data_loader("../train.csv").iloc[:, :-1]
    db_scan = DBSCAN(eps=eps, min_samples=min_pts)
    predicted_labels = db_scan.fit_predict(x)
    silhouette = silhouette_score(x, predicted_labels)
    print(f"Silhouette: {silhouette}")


def dbscan(dataset="raw", normalization_type="raw", eps=0.5, min_samples=3):
    if dataset == "normal":
        apply_dbscan_normal_dataset(normalization_type, eps, min_samples)
    elif dataset == "fe_resolution":
        apply_dbscan_fe_resolution_dataset(normalization_type, eps, min_samples)
    elif dataset == "fe_screen_size":
        apply_dbscan_fe_screen_size_dataset(normalization_type, eps, min_samples)
    elif dataset == "raw":
        apply_dbscan_raw_dataset(eps, min_samples)
    else:
        raise Exception("No such dataset!")

