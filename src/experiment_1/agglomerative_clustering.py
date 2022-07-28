import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from src.experiment_1.binning_utils import calculate_number_of_bins

dataset = pd.read_csv('../../resources/data/dataset_v1/dataset_1.1.csv')

norm_peaks = dataset['peak_magnitude_norm'].values
n_freedman = calculate_number_of_bins(norm_peaks, method='fd')

cluster = AgglomerativeClustering(n_clusters=n_freedman,
                                  affinity='euclidean', linkage='ward')
cluster.fit_predict(norm_peaks.reshape(-1, 1))
classes = 1 + cluster.labels_
dataset.insert(loc=6, column='bins_fd_agg', value=classes)

dataset.to_csv('../../resources/data/dataset_binned_v2.csv', sep=',', encoding='utf-8', index=False)
