import pandas as pd

from sklearn import metrics

dataset = pd.read_csv('../../../resources/data/cases/dataset_v1/dataset_1.4.csv')

norm_peaks = dataset['peak_magnitude_norm'].values.reshape(-1, 1)
km_clusters = dataset['cluster_fd_km'].values
agg_clusters = dataset['cluster_fd_agg'].values
fd_jenks_clusters = dataset['cluster_fd_jenks'].values
sturges_jenks_clusters = dataset['cluster_sturges_jenks'].values
scott_jenks_clusters = dataset['cluster_scott_jenks'].values

km_db = metrics.davies_bouldin_score(norm_peaks, km_clusters)
agg_db = metrics.davies_bouldin_score(norm_peaks, agg_clusters)
fd_jenks_db = metrics.davies_bouldin_score(norm_peaks, fd_jenks_clusters)
sturges_db = metrics.davies_bouldin_score(norm_peaks, sturges_jenks_clusters)
scott_db = metrics.davies_bouldin_score(norm_peaks, scott_jenks_clusters)

print('--Davies - Bouldin scores--')

print('FD + K-means: ' + str(km_db))
print('FD + Agglomerative: ' + str(agg_db))
print('FD + JF: ' + str(fd_jenks_db))
print('Sturges + JF: ' + str(sturges_db))
print('Scott + JF: ' + str(scott_db))
