import pandas as pd

from sklearn import metrics

dataset = pd.read_csv('../../../resources/data/cases/dataset_v2/dataset_2.3.csv')

norm_peaks = dataset['peak_magnitude_norm'].values.reshape(-1, 1)
fd_jenks_clusters = dataset['fd_jenks_clusters'].to_numpy()
sturges_jenks_clusters = dataset['sturges_jenks_clusters'].to_numpy()
scott_jenks_clusters = dataset['scott_jenks_clusters'].to_numpy()
agg_clusters = dataset['fd_agg_clusters'].to_numpy()
km_clusters = dataset['fd_km_clusters'].to_numpy()
uniform_clusters = dataset['fd_uniform_clusters'].to_numpy()

fd_jenks_db = metrics.davies_bouldin_score(norm_peaks, fd_jenks_clusters)
sturges_db = metrics.davies_bouldin_score(norm_peaks, sturges_jenks_clusters)
scott_db = metrics.davies_bouldin_score(norm_peaks, scott_jenks_clusters)
agg_db = metrics.davies_bouldin_score(norm_peaks, agg_clusters)
km_db = metrics.davies_bouldin_score(norm_peaks, km_clusters)
uniform_db = metrics.davies_bouldin_score(norm_peaks, uniform_clusters)

print('--Davies - Bouldin scores--')

print('FD + JF: ' + str(fd_jenks_db))
print('Sturges + JF: ' + str(sturges_db))
print('Scott + JF: ' + str(scott_db))
print('FD + Agglomerative: ' + str(agg_db))
print('FD + K-means: ' + str(km_db))
print('FD + Uniform binning: ' + str(uniform_db))
