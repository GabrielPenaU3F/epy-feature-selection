import pandas as pd

from sklearn import metrics

dataset = pd.read_csv('../../../resources/data/dataset_binned_v4.csv')

norm_peaks = dataset['peak_magnitude_norm'].values.reshape(-1, 1)
km_clusters = dataset['cluster_fd_km'].values
agg_clusters = dataset['cluster_fd_agg'].values
fd_jenks_clusters = dataset['cluster_fd_jenks'].values
sturges_jenks_clusters = dataset['cluster_sturges_jenks'].values
scott_jenks_clusters = dataset['cluster_scott_jenks'].values

km_sil = metrics.silhouette_score(norm_peaks, km_clusters, metric='euclidean')
agg_sil = metrics.silhouette_score(norm_peaks, agg_clusters, metric='euclidean')
fd_jenks_sil = metrics.silhouette_score(norm_peaks, fd_jenks_clusters, metric='euclidean')
sturges_sil = metrics.silhouette_score(norm_peaks, sturges_jenks_clusters, metric='euclidean')
scott_sil = metrics.silhouette_score(norm_peaks, scott_jenks_clusters, metric='euclidean')

print('--Silhouette scores--')

print('FD + K-means: ' + str(km_sil))
print('FD + Agglomerative: ' + str(agg_sil))
print('FD + JF: ' + str(fd_jenks_sil))
print('Sturges + JF: ' + str(sturges_sil))
print('Scott + JF: ' + str(scott_sil))
