import numpy as np
import pandas as pd

from src.binning.binning_utils import cut_dataset_into_clusters

dataset = pd.read_csv('../../resources/data/dataset_binned_v3.csv')

fd_agg_clustered = cut_dataset_into_clusters(dataset, 'bins_fd_agg')
fd_jenks_clustered = cut_dataset_into_clusters(dataset, 'bins_fd')
sturges_jenks_clustered = cut_dataset_into_clusters(dataset, 'bins_sturges')
scott_jenks_clustered = cut_dataset_into_clusters(dataset, 'bins_scott')

fd_agg_clustered.to_csv('../../resources/data/agg_fd_clustered_dataset.csv', sep=',', encoding='utf-8', index=False)
fd_jenks_clustered.to_csv('../../resources/data/jenks_fd_clustered_dataset.csv', sep=',', encoding='utf-8', index=False)
sturges_jenks_clustered.to_csv('../../resources/data/jenks_sturges_clustered_dataset.csv',
                               sep=',', encoding='utf-8', index=False)
scott_jenks_clustered.to_csv('../../resources/data/jenks_scott_clustered_dataset.csv',
                             sep=',', encoding='utf-8', index=False)
