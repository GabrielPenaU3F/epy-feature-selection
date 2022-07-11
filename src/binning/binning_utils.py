import numpy as np
import pandas as pd
from scipy import stats
import jenkspy

''' # Unused
def freedman_diaconis_binwidth(data):
    iqr = stats.iqr(data)
    n = len(data)
    return (2 * iqr) / np.power(n, 1/3)


def freedman_diaconis_bins(data):
    binwidth = freedman_diaconis_binwidth(data)
    return int((np.max(data) - np.min(data))/binwidth)
'''


def calculate_number_of_bins(data, method='fd'):
    n_of_edges = np.histogram_bin_edges(data, bins=method)
    return len(n_of_edges) - 1


def fisher_jenks_breaks(data, n_bins):
    return jenkspy.jenks_breaks(data, nb_class=n_bins)


def cut_data_in_column(df, data_col, breaks):
    n_bins = len(breaks) - 1
    labels = np.arange(1, n_bins + 1)
    return pd.cut(df[data_col], bins=breaks, labels=labels, include_lowest=True)


def cut_dataset_into_clusters(dataset, cluster_column_name):
    clusters = np.sort(dataset[cluster_column_name].unique())
    clustered_dataset = pd.DataFrame(columns=list(clusters))
    for cluster in clusters:
        cluster_content = dataset[dataset[cluster_column_name] == cluster]
        column = pd.DataFrame(columns=[cluster])
        for index, row in cluster_content.iterrows():
            location = row['location']
            n_peak = row['n_peak']
            point_title = str(location) + ' - peak Nº' + str(n_peak)
            column = column.append({cluster: point_title}, ignore_index=True)
        clustered_dataset[cluster] = column
    return clustered_dataset
