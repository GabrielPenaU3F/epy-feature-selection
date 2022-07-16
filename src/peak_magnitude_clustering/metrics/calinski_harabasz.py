import pandas as pd

from sklearn import metrics

dataset = pd.read_csv('../../../resources/data/dataset_binned_v4.csv')

norm_peaks = dataset['peak_magnitude_norm'].values.reshape(-1, 1)
km_clusters = dataset['cluster_fd_km'].values
agg_clusters = dataset['cluster_fd_agg'].values
fd_jenks_clusters = dataset['cluster_fd_jenks'].values
sturges_jenks_clusters = dataset['cluster_sturges_jenks'].values
scott_jenks_clusters = dataset['cluster_scott_jenks'].values

km_ch = metrics.calinski_harabasz_score(norm_peaks, km_clusters)
agg_ch = metrics.calinski_harabasz_score(norm_peaks, agg_clusters)
fd_jenks_ch = metrics.calinski_harabasz_score(norm_peaks, fd_jenks_clusters)
sturges_ch = metrics.calinski_harabasz_score(norm_peaks, sturges_jenks_clusters)
scott_ch = metrics.calinski_harabasz_score(norm_peaks, scott_jenks_clusters)

print('--Calinski - Harabasz scores--')

print('FD + K-means: ' + str(km_ch))
print('FD + Agglomerative: ' + str(agg_ch))
print('FD + JF: ' + str(fd_jenks_ch))
print('Sturges + JF: ' + str(sturges_ch))
print('Scott + JF: ' + str(scott_ch))
