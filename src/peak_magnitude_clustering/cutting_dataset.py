import pandas as pd

from src.peak_magnitude_clustering.binning_utils import cut_dataset_into_clusters

dataset = pd.read_csv('../../resources/data/dataset_clustered_v4.csv')

fd_km_clustered = cut_dataset_into_clusters(dataset, 'cluster_fd_km')
fd_agg_clustered = cut_dataset_into_clusters(dataset, 'cluster_fd_agg')
fd_jenks_clustered = cut_dataset_into_clusters(dataset, 'cluster_fd')
sturges_jenks_clustered = cut_dataset_into_clusters(dataset, 'cluster_sturges')
scott_jenks_clustered = cut_dataset_into_clusters(dataset, 'cluster_scott')

fd_km_clustered.to_csv('../../resources/data/km_fd_clustered_dataset.csv', sep=',', encoding='utf-8', index=False)
fd_agg_clustered.to_csv('../../resources/data/agg_fd_clustered_dataset.csv', sep=',', encoding='utf-8', index=False)
fd_jenks_clustered.to_csv('../../resources/data/jenks_fd_clustered_dataset.csv', sep=',', encoding='utf-8', index=False)
sturges_jenks_clustered.to_csv('../../resources/data/jenks_sturges_clustered_dataset.csv',
                               sep=',', encoding='utf-8', index=False)
scott_jenks_clustered.to_csv('../../resources/data/jenks_scott_clustered_dataset.csv',
                             sep=',', encoding='utf-8', index=False)
