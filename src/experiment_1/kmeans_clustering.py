import pandas as pd
from sklearn.cluster import KMeans

from src.experiment_1.binning_utils import calculate_number_of_bins

dataset = pd.read_csv('../../resources/data/cases/dataset_v1/dataset_1.3.csv')

norm_peaks = dataset['peak_magnitude_norm'].values
n_freedman = calculate_number_of_bins(norm_peaks, method='fd')

cluster = KMeans(n_freedman)
cluster.fit_predict(norm_peaks.reshape(-1, 1))
classes = 1 + cluster.labels_
dataset.insert(loc=6, column='cluster_fd_km', value=classes)

dataset.to_csv('../../resources/data/dataset_binned_v4.csv', sep=',', encoding='utf-8', index=False)
