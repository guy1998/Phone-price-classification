from Data_Manipulation.normalization import data_loader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def apply_k_means_fe_resolution_dataset(normalization_type):
    x = data_loader("Datasets/fe_resolution_" + normalization_type + "_dataset.csv").iloc[:, :-1]
    kmeans = KMeans(n_clusters=4, random_state=38, n_init=10)
    kmeans.fit(x)
    predicted_labels = kmeans.labels_
    silhouette = silhouette_score(x, predicted_labels)
    print(f"Silhouette: {silhouette}")


def apply_k_means_normal_dataset(normalization_type):
    x = data_loader("Datasets/" + normalization_type + "_dataset.csv").iloc[:, :-1]
    kmeans = KMeans(n_clusters=4, random_state=38, n_init=10)
    kmeans.fit(x)
    predicted_labels = kmeans.labels_
    silhouette = silhouette_score(x, predicted_labels)
    print(f"Silhouette: {silhouette}")


def apply_k_means_fe_screen_size_dataset(normalization_type):
    x = data_loader("Datasets/fe_screen_size_" + normalization_type + "_dataset.csv").iloc[:, :-1]
    kmeans = KMeans(n_clusters=4, random_state=38, n_init=10)
    kmeans.fit(x)
    predicted_labels = kmeans.labels_
    silhouette = silhouette_score(x, predicted_labels)
    print(f"Silhouette: {silhouette}")


def apply_k_means_raw_dataset():
    x = data_loader("../train.csv").iloc[:, :-1]
    kmeans = KMeans(n_clusters=4, random_state=38, n_init=10)
    kmeans.fit(x)
    predicted_labels = kmeans.labels_
    silhouette = silhouette_score(x, predicted_labels)
    print(f"Silhouette: {silhouette}")


def k_means(dataset="raw", normalization_type="raw"):
    if dataset == "normal":
        apply_k_means_normal_dataset(normalization_type)
    elif dataset == "fe_resolution":
        apply_k_means_fe_resolution_dataset(normalization_type)
    elif dataset == "fe_screen_size":
        apply_k_means_fe_screen_size_dataset(normalization_type)
    elif dataset == "raw":
        apply_k_means_raw_dataset()
    else:
        raise Exception("No such dataset!")

