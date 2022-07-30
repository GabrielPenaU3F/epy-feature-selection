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

fd_jenks_ch = metrics.calinski_harabasz_score(norm_peaks, fd_jenks_clusters)
sturges_ch = metrics.calinski_harabasz_score(norm_peaks, sturges_jenks_clusters)
scott_ch = metrics.calinski_harabasz_score(norm_peaks, scott_jenks_clusters)
agg_ch = metrics.calinski_harabasz_score(norm_peaks, agg_clusters)
km_ch = metrics.calinski_harabasz_score(norm_peaks, km_clusters)
uniform_ch = metrics.calinski_harabasz_score(norm_peaks, uniform_clusters)

print('--Calinski - Harabasz scores--')

print('FD + JF: ' + str(fd_jenks_ch))
print('Sturges + JF: ' + str(sturges_ch))
print('Scott + JF: ' + str(scott_ch))
print('FD + Agglomerative: ' + str(agg_ch))
print('FD + K-means: ' + str(km_ch))
print('FD + Uniform binning: ' + str(uniform_ch))
