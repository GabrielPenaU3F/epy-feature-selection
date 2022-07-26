import pandas as pd

from sklearn import metrics

dataset = pd.read_csv('../../../../resources/data/dataset_v2/dataset_2.3.csv')

norm_peaks = dataset['peak_magnitude_norm'].values.reshape(-1, 1)
fd_jenks_clusters = dataset['fd_jenks_clusters'].to_numpy()
sturges_jenks_clusters = dataset['sturges_jenks_clusters'].to_numpy()
scott_jenks_clusters = dataset['scott_jenks_clusters'].to_numpy()
agg_clusters = dataset['fd_agg_clusters'].to_numpy()
km_clusters = dataset['fd_km_clusters'].to_numpy()
uniform_clusters = dataset['fd_uniform_clusters'].to_numpy()

fd_jenks_sil = metrics.silhouette_score(norm_peaks, fd_jenks_clusters, metric='euclidean')
sturges_sil = metrics.silhouette_score(norm_peaks, sturges_jenks_clusters, metric='euclidean')
scott_sil = metrics.silhouette_score(norm_peaks, scott_jenks_clusters, metric='euclidean')
agg_sil = metrics.silhouette_score(norm_peaks, agg_clusters, metric='euclidean')
km_sil = metrics.silhouette_score(norm_peaks, km_clusters, metric='euclidean')
uniform_sil = metrics.silhouette_score(norm_peaks, uniform_clusters, metric='euclidean')

print('--Silhouette scores--')

print('FD + JF: ' + str(fd_jenks_sil))
print('Sturges + JF: ' + str(sturges_sil))
print('Scott + JF: ' + str(scott_sil))
print('FD + Agglomerative: ' + str(agg_sil))
print('FD + K-means: ' + str(km_sil))
print('FD + Uniform binning: ' + str(uniform_sil))
