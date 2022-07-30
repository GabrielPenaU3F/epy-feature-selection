# INPUT: dataset_deaths_2.1.csv
# OUTPUT: dataset_deaths_2.2.csv

# Group the normalized peak magnitude into clusters or bins
# We generate three output columns using uniform binning, K-Means, and agglomerative clustering,
# with the number of classes chosen according to the Freedman-Diaconis method
# We also generate another three output columns by using Jenks-Fisher breaks,
# with the number of classes chosen according to Freedman-Diaconis, Sturges and Scott methods
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import KBinsDiscretizer

from src.experiment_1.binning_utils import calculate_number_of_bins, fisher_jenks_breaks, \
    cut_data_in_column

dataset = pd.read_csv('../../../../resources/data/deaths/dataset_deaths_2.1.csv')
norm_peaks = dataset['peak_magnitude_norm'].to_numpy()

# Obtaining the number of classes
n_freedman = calculate_number_of_bins(norm_peaks, method='fd')
n_sturges = calculate_number_of_bins(norm_peaks, method='sturges')
n_scott = calculate_number_of_bins(norm_peaks, method='scott')

# Jenks-Fisher method

breaks_freedman = fisher_jenks_breaks(norm_peaks, n_freedman)
breaks_sturges = fisher_jenks_breaks(norm_peaks, n_sturges)
breaks_scott = fisher_jenks_breaks(norm_peaks, n_scott)

fd_jf_clusters = cut_data_in_column(dataset, 'peak_magnitude_norm', breaks_freedman)
sturges_jf_clusters = cut_data_in_column(dataset, 'peak_magnitude_norm', breaks_sturges)
scott_jf_clusters = cut_data_in_column(dataset, 'peak_magnitude_norm', breaks_scott)

# Agglomerative Clustering

cluster = AgglomerativeClustering(n_clusters=n_freedman, affinity='euclidean', linkage='ward')
cluster.fit_predict(norm_peaks.reshape(-1, 1))
fd_agg_clusters = 1 + cluster.labels_

# K-Means

cluster = KMeans(n_freedman)
cluster.fit_predict(norm_peaks.reshape(-1, 1))
fd_km_clusters = 1 + cluster.labels_

# Uniform binning

kbins = KBinsDiscretizer(n_bins=n_freedman, encode='ordinal', strategy='uniform')
fd_kb_clusters = 1 + kbins.fit_transform(norm_peaks.reshape(-1, 1))

# Updating the dataset

dataset.insert(loc=3, column='fd_jenks_clusters', value=fd_jf_clusters.astype('int64'))
dataset.insert(loc=4, column='sturges_jenks_clusters', value=sturges_jf_clusters.astype('int64'))
dataset.insert(loc=5, column='scott_jenks_clusters', value=scott_jf_clusters.astype('int64'))
dataset.insert(loc=6, column='fd_agg_clusters', value=fd_agg_clusters.astype('int64'))
dataset.insert(loc=7, column='fd_km_clusters', value=fd_km_clusters.astype('int64'))
dataset.insert(loc=8, column='fd_uniform_clusters', value=fd_kb_clusters.astype('int64'))

# Out
dataset.to_csv('../../../../resources/data/deaths/dataset_deaths_2.2.csv', sep=',', encoding='utf-8', index=False)
